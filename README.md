cat > README.md <<'EOF'

\# qsr-pipeline-ci



Pipeline for pulling data from Domo, enriching/aggregating to store-day outputs, and joining keymetrics.



\## Local run

From repo root:



```bash

cd src

python run\_full\_pipeline\_v2.py --help



