#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -x .venv/bin/python ]; then
  echo 'Ambiente não encontrado. Execute: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
  exit 1
fi
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 7860
