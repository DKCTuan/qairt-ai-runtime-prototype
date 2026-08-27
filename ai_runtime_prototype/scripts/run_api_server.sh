#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
exec python3 tools/model_deploy_server.py "$@"
