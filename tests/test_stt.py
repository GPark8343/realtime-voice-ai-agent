"""SonioxSession 테스트 — 가짜 Soniox 연결(FakeWS)로 응답 해석과 장애 처리를 확인한다. 네트워크·키 필요 없음."""
import asyncio
import json

import websockets

import stt


class FakeWS:
    """Soniox WebSocket 흉내. messages 를 차례로 내주고, close_error 면 마지막에 비정상 종료를 던진다."""

    def __init__(self, messages=(), close_error=False, fail_send=False):
        self.messages = [m if isinstance(m, str) else json.dumps(m, ensure_ascii=False) for m in messages]
        self.close_error = close_error
        self.fail_send = fail_send
        self.sent = []

    async def send(self, data):
        if self.fail_send:
            raise websockets.ConnectionClosedError(None, None)
        self.sent.append(data)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self.messages:
            yield m
        if self.close_error:
            raise websockets.ConnectionClosedError(None, None)

    async def close(self):
        pass


def tok(text, final=True):
    return {"text": text, "is_final": final}


def run_reader(ws):
    s = stt.SonioxSession(api_key="test")
    s.ws = ws
    asyncio.run(s._reader())
    return s


# ── 응답 해석 ──────────────────────────────────────────────

def test_end_token_sets_end_event_and_keeps_final_text():
    s = run_reader(FakeWS([{"tokens": [tok("배송"), tok(" 언제요"), tok("<end>")]}]))
    assert s.end_event.is_set()
    assert s.text == "배송 언제요"


def test_partial_text_keeps_all_tokens_of_message():
    """한 응답 안의 미확정 토큰 여러 개를 모두 이어 붙여야 한다 (마지막 토큰만 남기면 말이 잘린다)."""
    s = run_reader(FakeWS([{"tokens": [tok("주문", final=False), tok("번호요", final=False)]}], close_error=False))
    assert s.text == "주문번호요"


# ── 장애 처리: Soniox 가 죽어도 '말 끝'으로 착각하지 않는다 ──

def test_error_response_marks_failed_not_end():
    s = run_reader(FakeWS([{"tokens": [], "error_code": 401, "error_type": "unauthenticated", "error_message": "bad key"}]))
    assert s.failed
    assert not s.end_event.is_set()


def test_abnormal_close_before_end_marks_failed_not_end():
    s = run_reader(FakeWS([{"tokens": [tok("배송", final=False)]}], close_error=True))
    assert s.failed
    assert not s.end_event.is_set()


def test_stream_ending_without_end_marks_failed_not_end():
    s = run_reader(FakeWS([{"tokens": [tok("배송")]}]))
    assert s.failed
    assert not s.end_event.is_set()


def test_feed_after_send_failure_does_not_raise():
    s = stt.SonioxSession(api_key="test")
    s.ws = FakeWS(fail_send=True)
    asyncio.run(s.feed(b"\0" * 640))       # 예외가 나면 server.forward_audio 태스크가 죽는다
    assert s.failed


def test_connect_failure_marks_failed_without_raising(monkeypatch):
    async def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(stt.websockets, "connect", boom)
    s = stt.SonioxSession(api_key="test")
    asyncio.run(s.connect())                 # 예외가 나면 handle_turn 전체가 죽어 소리 없이 끝난다
    assert s.failed


# ── 설정: 환경 변수로 말 끝 조절값·용어 목록을 켤 수 있다 (기본은 기존과 같음) ──

def connect_and_get_config(monkeypatch):
    fake = FakeWS()

    async def fake_connect(*a, **k):
        return fake
    monkeypatch.setattr(stt.websockets, "connect", fake_connect)

    async def go():
        s = stt.SonioxSession(api_key="test")
        await s.connect()
        if s._reader_task:
            await s._reader_task
    asyncio.run(go())
    return json.loads(fake.sent[0])


def test_connect_default_config_unchanged(monkeypatch):
    for k in ("SONIOX_EP_LEVEL", "SONIOX_EP_SENS", "SONIOX_EP_MAX_MS", "SONIOX_CONTEXT_TERMS"):
        monkeypatch.delenv(k, raising=False)
    cfg = connect_and_get_config(monkeypatch)
    assert cfg["model"] == "stt-rt-v6" and cfg["enable_endpoint_detection"] is True
    assert "endpoint_latency_adjustment_level" not in cfg
    assert "context" not in cfg


def test_connect_endpoint_params_from_env(monkeypatch):
    monkeypatch.setenv("SONIOX_EP_LEVEL", "2")
    monkeypatch.setenv("SONIOX_EP_SENS", "0.3")
    monkeypatch.setenv("SONIOX_EP_MAX_MS", "1500")
    cfg = connect_and_get_config(monkeypatch)
    assert cfg["endpoint_latency_adjustment_level"] == 2
    assert cfg["endpoint_sensitivity"] == 0.3
    assert cfg["max_endpoint_delay_ms"] == 1500


def test_connect_uses_runtime_soniox_url(monkeypatch):
    """연결 주소는 세션을 열 때 _soniox_url() 로 정한다 (SONIOX_WS_URL > BAZE 중계 > 직접 Soniox)."""
    monkeypatch.setenv("SONIOX_WS_URL", "wss://example.test/soniox/transcribe-websocket")
    seen = {}

    async def fake_connect(url, *a, **k):
        seen["url"] = url
        return FakeWS()
    monkeypatch.setattr(stt.websockets, "connect", fake_connect)

    s = stt.SonioxSession(api_key="test")
    asyncio.run(s.connect())
    assert seen["url"] == "wss://example.test/soniox/transcribe-websocket"
    assert not (s.failed and "연결 실패" in (s.error or ""))


def test_connect_context_terms_when_enabled(monkeypatch):
    monkeypatch.setenv("SONIOX_CONTEXT_TERMS", "1")
    cfg = connect_and_get_config(monkeypatch)
    assert "하루온" in cfg["context"]["terms"]
