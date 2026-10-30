#!/bin/bash
# Local contract check: start the service on 127.0.0.1, then Add / duplicate Add / Search / other user / no auth.
# Needs DASHSCOPE_API_KEY (the Add embeds two short messages).
set -eu
cd "$(dirname "$0")/../service"
SP=$(mktemp -d)
AML_TOKEN=testtok AML_DB=$SP/mem.sqlite AML_CACHE=$SP/emb.sqlite uvicorn server:app --port 8765 --log-level warning > "$SP/srv.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$SP"' EXIT
U=http://127.0.0.1:8765
for i in $(seq 1 30); do curl -s -m 2 "$U/health" >/dev/null && break; sleep 1; done
J='Content-Type: application/json'
ADD='{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","timestamp":1685000000000,"content":"I moved to Hangzhou last week for a new job."},{"role":"assistant","timestamp":1685000001000,"content":"Congrats on the move!"}]}'
curl -s -m 60 -X POST -H "$J" -H 'Authorization: Token testtok' -d "$ADD" "$U/add"; echo
curl -s -m 60 -X POST -H "$J" -H 'Authorization: Token testtok' -d "$ADD" "$U/add"; echo
curl -s -m 60 -X POST -H "$J" -H 'X-Api-Key: testtok' -d '{"query":"Where does the user live now?","user_id":"u1","top_k":5}' "$U/search"; echo
curl -s -m 60 -X POST -H "$J" -H 'Authorization: Bearer testtok' -d '{"query":"x","user_id":"nobody","top_k":5}' "$U/search"; echo
curl -s -m 60 -o /dev/null -w 'no auth -> %{http_code}\n' -X POST -H "$J" -d '{"query":"x","user_id":"u1"}' "$U/search"
