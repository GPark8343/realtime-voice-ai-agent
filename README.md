# Realtime Voice AI Customer Service Agent

> GDGoC Korea University × GenerativeLab 해커톤(DIRECT:ON 2026) 4인 팀 프로젝트에서 제가 담당한 파트(`agent_server/`)만 발췌한 레포입니다.
> 해커톤에서 제공된 실제 브랜드/고객 데이터는 규정상 공개할 수 없어 전부 제외했고, `data/`는 코드가 그대로 동작하도록 직접 새로 만든 가상 예시 데이터입니다.

## 무엇을 만들었나

전화로 들어오는 손님 음성을 실시간으로 받아, STT → LLM(함수 호출) → TTS 파이프라인으로 자동 응대하는 WebSocket 서버입니다. 한 번의 연결 = 손님 발화 1번 = 응대 1번 구조이며, 손님이 주문 조회·반품 접수·상품 문의 등을 요청하면 실제 쇼핑몰 시스템을 흉내 낸 모의 함수 서버(`mock_tools/`)를 호출해 결과를 확인한 뒤 답합니다.

```
손님 음성 → STT(Soniox) → LLM 함수 호출 판단(GPT) → 함수 실행 → TTS(Gemini) → 응답 음성
```

## 기술적으로 풀었던 문제들

- **TTS가 실제로는 스트리밍이 아님**: API는 스트리밍 형식이지만 백엔드가 전체 텍스트를 다 합성한 뒤에야 응답을 내려준다는 걸 실측(TTFB 2.5~6초)으로 확인하고, (1) 자주 쓰는 맞장구·대기 멘트를 사전 음성 파일로 캐싱(`agent_server/cues.py`), (2) 답변을 문장 단위로 쪼개 동시에 합성해 먼저 끝난 문장부터 송출(`agent_server/server.py`의 `speak()`)하는 방식으로 체감 지연을 줄였습니다.
- **LLM 게이트웨이의 개인정보 마스킹이 함수 호출을 깨뜨리는 문제**: 손님 발화 속 전화번호가 LLM 응답에서 `[PHONE]` 같은 플레이스홀더로 치환되는 현상을 실측으로 발견 → 긴 숫자열을 `<NUM1>` 토큰으로 바꿔 LLM에 넘기고, 실제 함수 호출 직전에만 원래 숫자로 복원하는 방식으로 우회(`agent_server/llm.py`의 `mask_digit_runs`/`unmask_value`).
- **말 끝 감지의 이중 안전장치**: 실시간 STT의 자체 종료 판정(Soniox `<end>`)이 가끔 안 올 때를 대비해 RMS 기반 무음 타임아웃을 보조 신호로 병행.
- **끼어들기(barge-in) 인지형 응답**: 맞장구를 보내기 직전 손님이 다시 말하기 시작했으면 건너뛰는 안전장치(`VoiceTracker`/`send_ack`).
- **크레딧 0 테스트**: 실제 API를 전혀 호출하지 않는 가짜 STT/LLM/TTS로 파이프라인 로직만 격리해서 검증하는 테스트 구조(`tests/`).

## 실행

```bash
pip install -r requirements.txt
bash start_mock.sh data/mock_db.json   # 모의 함수 서버, :8100
bash start_server.sh                    # 상담 서버, :9000
pytest tests/ -q
```

## 팀 프로젝트 안내

4인 팀(`GDG on Campus Korea University` 소속)으로 진행한 해커톤 과제였고, 이 레포는 그중 제가 구현한 실시간 음성 파이프라인(`agent_server/`, `mock_tools/`)만 담았습니다. STT/LLM/TTS 연동, 모의 함수 서버 설계, 프롬프트 규칙 설계, 버그 발견·수정, 테스트 작성을 담당했습니다.
