#!/usr/bin/env bash
# 상담 서버 시작 — 최종 측정 때 운영진이 이 파일을 그대로 실행한다. 9000번 포트에서 WebSocket을 받으면 된다.
cd "$(dirname "$0")"
exec python3 agent_server/server.py --port 9000
