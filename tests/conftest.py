# 테스트에서 mock_tools/, agent_server/ 의 모듈을 import 할 수 있게 경로를 잡는다
# (agent_server/server.py 가 `import stt` 처럼 폴더 기준으로 import 하므로 같은 방식을 따른다)
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "mock_tools"))
sys.path.insert(0, os.path.join(ROOT, "agent_server"))
