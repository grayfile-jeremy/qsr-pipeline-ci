#!/usr/bin/env python3
"""
domo_query_std_filters.py — Generalized Domo Query API runner with flexible, column-based filters.

Examples:
  python3 domo_query_std_filters.py \
    --dataset-id 11111111-2222-3333-4444-555555555555 \
    --out filtered.csv \
    --filter "Store# is 07-0002,107-0001" \
    --filter "Segment is Store" \
    --filter "NA_Code is not EBITDA,EBITDA - 4 Wall,EBITDA - 4 Wall 2,EBITDAR - 4 Wall,EBITDAR - 4 Wall 2,Store Count" \
    --filter "Timing is not YTD,Full Year"

More forms (mix & match):
  --filter "Revenue >= 100000"
  --filter "Closed Date between 2025-01-01..2025-12-31"
  --filter "Region like %West%"
  --filter "Region not like %Test%"
  --filter "Notes is null"
  --filter "Notes is not null"
  --filter "Category in A,B,C"        # same as 'is'
  --filter "Category not in X,Y,Z"    # same as 'is not'

Notes:
- Multiple --filter flags are AND-combined.
- Inside a single 'is / in' or 'is not / not in' filter, the values list is OR-combined via IN / NOT IN.
- Identifiers are safely quoted; string literals are quoted.
- Auth via env: DOMO_CLIENT_ID, DOMO_CLIENT_SECRET
"""

import os, sys, time, argparse, requests, json, csv, re
from typing import Iterable, List, Tuple, Optional
from requests.auth import HTTPBasicAuth

DOMO_BASE = "https://api.domo.com"
OAUTH_URL  = f"{DOMO_BASE}/oauth/token"

# ---------------------------
# Auth
# ---------------------------
def _get_oauth_token() -> str:
    cid  = os.environ.get("DOMO_CLIENT_ID")
    csec = os.environ.get("DOMO_CLIENT_SECRET")
    if not cid or not csec:
        print("[ERR] DOMO_CLIENT_ID and DOMO_CLIENT_SECRET must be set.", file=sys.stderr)
        sys.exit(2)
    r = requests.post(
        OAUTH_URL,
        params={"grant_type": "client_credentials", "scope": "data"},
        auth=HTTPBasicAuth(cid, csec),
        timeout=30
    )
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERR] OAuth failed: {e} — {r.text}", file=sys.stderr)
        sys.exit(2)
    return r.json()["access_token"]

# ---------------------------
# SQL helpers
# ---------------------------
def q_ident(col: str) -> str:
    col = col.strip()
    return '"' + col.replace('"', '""') + '"'

def q_literal(val: str) -> str:
    # Always quote as string to avoid locale/type surprises; Domo casts as needed.
    return "'" + val.replace("'", "''") + "'"

def _split_values_list(vs: str) -> List[str]:
    # Split on commas not considering quotes (simple + robust for our use)
    return [v.strip() for v in vs.split(",") if v.strip() != ""]

def in_list_sql(values: Iterable[str]) -> str:
    items = [q_literal(v) for v in values]
    if not items:
        return "(NULL)"
    return "(" + ",".join(items) + ")"

# ---------------------------
# Filter parsing
# ---------------------------
_OP_PATTERNS = [
    # Two-word / multi-word operators first
    (re.compile(r"^\s*(?P<col>.+?)\s+is\s+not\s+(?P<vals>.+)$", re.I), "notin"),
    (re.compile(r"^\s*(?P<col>.+?)\s+not\s+in\s+(?P<vals>.+)$", re.I), "notin"),
    (re.compile(r"^\s*(?P<col>.+?)\s+is\s+null\s*$", re.I), "isnull"),
    (re.compile(r"^\s*(?P<col>.+?)\s+is\s+not\s+null\s*$", re.I), "notnull"),
    (re.compile(r"^\s*(?P<col>.+?)\s+not\s+like\s+(?P<val>.+)$", re.I), "notlike"),
    (re.compile(r"^\s*(?P<col>.+?)\s+between\s+(?P<a>.+?)\s*\.\.\s*(?P<b>.+)$", re.I), "between"),
    (re.compile(r"^\s*(?P<col>.+?)\s+between\s+(?P<a>.+?)\s+to\s+(?P<b>.+)$", re.I), "between"),
    # Single-word / symbol operators
    (re.compile(r"^\s*(?P<col>.+?)\s+is\s+(?P<vals>.+)$", re.I), "in"),
    (re.compile(r"^\s*(?P<col>.+?)\s+in\s+(?P<vals>.+)$", re.I), "in"),
    (re.compile(r"^\s*(?P<col>.+?)\s*>=\s*(?P<val>.+)$", re.I), "gte"),
    (re.compile(r"^\s*(?P<col>.+?)\s*<=\s*(?P<val>.+)$", re.I), "lte"),
    (re.compile(r"^\s*(?P<col>.+?)\s*>\s*(?P<val>.+)$", re.I), "gt"),
    (re.compile(r"^\s*(?P<col>.+?)\s*<\s*(?P<val>.+)$", re.I), "lt"),
    (re.compile(r"^\s*(?P<col>.+?)\s*!=\s*(?P<val>.+)$", re.I), "ne"),
    (re.compile(r"^\s*(?P<col>.+?)\s*<>\s*(?P<val>.+)$", re.I), "ne"),
    (re.compile(r"^\s*(?P<col>.+?)\s*=\s*(?P<val>.+)$", re.I), "eq"),
    (re.compile(r"^\s*(?P<col>.+?)\s+eq\s+(?P<val>.+)$", re.I), "eq"),
    (re.compile(r"^\s*(?P<col>.+?)\s+ne\s+(?P<val>.+)$", re.I), "ne"),
    (re.compile(r"^\s*(?P<col>.+?)\s+like\s+(?P<val>.+)$", re.I), "like"),
]

def parse_filter(expr: str) -> Optional[str]:
    """
    Turn a human-friendly filter expression into a SQL clause.
    Supported:
      - "<col> is v1,v2,v3"            -> "col" IN ('v1','v2','v3')
      - "<col> is not v1,v2"           -> "col" NOT IN ('v1','v2')
      - "<col> in v1,v2"               -> same as 'is'
      - "<col> not in v1,v2"           -> same as 'is not'
      - "<col> between a..b"           -> "col" BETWEEN 'a' AND 'b'
      - "<col> >= 10", "<col> = 5", "<col> != foo"
      - "<col> like %foo%", "<col> not like foo%"
      - "<col> is null", "<col> is not null"
    """
    s = expr.strip()
    if not s:
        return None

    # Allow colon form too: "Column:op:values"
    if ":" in s and s.count(":") >= 1:
        parts = [p.strip() for p in s.split(":", 2)]
        if len(parts) == 3:
            col, op, vals = parts
            return parse_filter(f"{col} {op} {vals}")
        elif len(parts) == 2:
            # e.g., "Notes:is null"
            col, op = parts
            return parse_filter(f"{col} {op}")

    for pat, tag in _OP_PATTERNS:
        m = pat.match(s)
        if not m:
            continue

        if tag in ("in", "notin"):
            col = q_ident(m.group("col"))
            vals = _split_values_list(m.group("vals"))
            if not vals:
                return None
            op = "IN" if tag == "in" else "NOT IN"
            return f"{col} {op} {in_list_sql(vals)}"

        if tag in ("eq","ne","gt","lt","gte","lte","like","notlike"):
            col = q_ident(m.group("col"))
            val = m.group("val").strip().strip('"').strip("'")
            sql_op = {
                "eq": "=", "ne": "<>", "gt": ">", "lt": "<",
                "gte": ">=", "lte": "<=", "like": "LIKE", "notlike": "NOT LIKE",
            }[tag]
            return f"{col} {sql_op} {q_literal(val)}"

        if tag == "between":
            col = q_ident(m.group("col"))
            a = m.group("a").strip().strip('"').strip("'")
            b = m.group("b").strip().strip('"').strip("'")
            return f"{col} BETWEEN {q_literal(a)} AND {q_literal(b)}"

        if tag == "isnull":
            col = q_ident(m.group("col"))
            return f"{col} IS NULL"

        if tag == "notnull":
            col = q_ident(m.group("col"))
            return f"{col} IS NOT NULL"

    # If no pattern matched, try last-resort: <col> is <vals>
    m = re.match(r"^\s*(?P<col>.+?)\s+is\s+(?P<vals>.+)$", s, flags=re.I)
    if m:
        col = q_ident(m.group("col"))
        vals = _split_values_list(m.group("vals"))
        if not vals:
            return None
        return f"{col} IN {in_list_sql(vals)}"

    print(f"[WARN] Unrecognized filter syntax: {expr!r}", file=sys.stderr)
    return None

# ---------------------------
# Query API call (JSON -> CSV fallback)
# ---------------------------
def query_to_csv(dataset_id: str, sql: str, out_path: str, accept: str = "application/json") -> int:
    token = _get_oauth_token()
    url = f"{DOMO_BASE}/v1/datasets/query/execute/{dataset_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,  # use application/json to avoid 406
        "Content-Type": "application/json"
    }
    body = {"sql": sql}

    with requests.post(url, headers=headers, json=body, stream=True, timeout=600) as resp:
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            msg = resp.text
            try:
                j = resp.json()
                msg = json.dumps(j, indent=2)
            except Exception:
                pass
            print(f"[ERR] Query API failed: {e}\n{msg}", file=sys.stderr)
            sys.exit(4)

        ctype = resp.headers.get("Content-Type", "").lower()

        if "text/csv" in ctype:
            written = 0
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk); written += len(chunk)
            return written

        data = resp.json()
        cols = [c["name"] if isinstance(c, dict) and "name" in c else str(c) for c in data.get("columns", [])]
        rows = data.get("rows", [])
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if cols: w.writerow(cols)
            for r in rows:
                w.writerow(r if isinstance(r, list) else list(r))
        return os.path.getsize(out_path)

# ---------------------------
# SQL assembly
# ---------------------------
def build_sql(select_cols: Optional[List[str]], filters: List[str], order_by: Optional[str], limit: Optional[int], offset: Optional[int]) -> str:
    sel = "*"
    if select_cols:
        sel = ", ".join(q_ident(c) for c in select_cols)

    where_parts = []
    for f in filters:
        clause = parse_filter(f)
        if clause:
            where_parts.append(clause)
        else:
            print(f"[WARN] Skipping filter (unparsed): {f}", file=sys.stderr)

    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

    order_sql = ""
    if order_by:
        # Example: "Closed Date desc, Store# asc"
        # We allow multiple, comma-separated items; each item may end with ASC/DESC.
        items = []
        for part in order_by.split(","):
            p = part.strip()
            if not p: continue
            m = re.match(r"^(?P<col>.+?)\s+(?P<dir>asc|desc)$", p, flags=re.I)
            if m:
                items.append(f"{q_ident(m.group('col'))} {m.group('dir').upper()}")
            else:
                items.append(f"{q_ident(p)}")
        if items:
            order_sql = " ORDER BY " + ", ".join(items)

    limit_sql = f" LIMIT {int(limit)}" if (limit is not None) else ""
    offset_sql = f" OFFSET {int(offset)}" if (offset is not None and limit is not None) else ""

    # Domo Query API uses 'table' as the dataset name
    return f"SELECT {sel} FROM table{where_sql}{order_sql}{limit_sql}{offset_sql}"

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Run flexible, server-side filters against a Domo dataset and save CSV.")
    ap.add_argument("--dataset-id", required=True, help="Domo dataset GUID")
    ap.add_argument("--out", required=True, help="Output CSV file path")

    # Flexible filter input
    ap.add_argument("--filter", action="append", default=[], help="Filter clause, e.g. \"Store# is 07-0002,107-0001\". Repeatable.")

    # Optional shaping
    ap.add_argument("--select", default="", help="Comma-separated columns to SELECT; default is all columns.")
    ap.add_argument("--order-by", default="", help="Comma-separated order list, e.g. \"Date desc, Store#\"")
    ap.add_argument("--limit", type=int, default=None, help="LIMIT rows")
    ap.add_argument("--offset", type=int, default=None, help="OFFSET rows (use with --limit)")

    args = ap.parse_args()

    select_cols = [c.strip() for c in args.select.split(",") if c.strip()] if args.select else None
    filters = args.filter or []
    order_by = args.order_by or None

    print(f"[INFO] Dataset: {args.dataset_id}")
    if select_cols:
        print(f"[INFO] Selecting columns: {select_cols}")
    print(f"[INFO] Filters ({len(filters)}): {filters}")
    if order_by:
        print(f"[INFO] Order by: {order_by}")
    if args.limit is not None:
        print(f"[INFO] Limit: {args.limit}  Offset: {args.offset}")

    sql = build_sql(select_cols, filters, order_by, args.limit, args.offset)

    print("[DEBUG] Generated SQL:")
    print(sql)

    t0 = time.time()
    bytes_written = query_to_csv(dataset_id=args.dataset_id, sql=sql, out_path=args.out, accept="application/json")
    dt = time.time() - t0
    kb = bytes_written / 1024.0
    print(f"[DONE] Wrote ~{kb:.1f} KiB to {args.out} in {dt:.2f}s (server-side filtered).")

if __name__ == "__main__":
    main()
