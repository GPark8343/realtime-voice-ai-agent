"""Soniox 실시간 STT(stt-rt-v6) WebSocket 클라이언트.

행사 지원 LLM 게이트웨이(BAZE)가 Soniox 실시간 접속도 중계한다 — 직접 Soniox 도메인이 아니라
`{OPENAI_BASE_URL 을 wss:// 로 바꾼 주소}/soniox/transcribe-websocket` 로 접속하고, 인증도 같은
OPENAI_API_KEY(BAZE 키)를 Bearer 로 쓴다. 실제 오디오로 접속·받아쓰기·<end> 토큰까지 확인됨.
직접 Soniox 계정(SONIOX_API_KEY/SONIOX_WS_URL)이 따로 생기면 그쪽을 우선한다.
"""
import asyncio
import json
import os
import sys

import websockets

RATE = 16000

# 말 끝 감지 조절값 (docs.soniox.com/stt/rt/endpoint-detection). 환경 변수를 주지 않으면 Soniox 기본값을 쓴다.
#   SONIOX_EP_LEVEL  → endpoint_latency_adjustment_level 0~3 (높을수록 빨리 끊음, 기본 0)
#   SONIOX_EP_SENS   → endpoint_sensitivity -1.0~1.0 (높을수록 쉽게 말 끝으로 봄, 기본 0.0)
#   SONIOX_EP_MAX_MS → max_endpoint_delay_ms 500~3000 (말이 끝난 뒤 이 안에는 반드시 <end>, 기본 2000)
ENDPOINT_ENV = (("SONIOX_EP_LEVEL", "endpoint_latency_adjustment_level", int),
                ("SONIOX_EP_SENS", "endpoint_sensitivity", float),
                ("SONIOX_EP_MAX_MS", "max_endpoint_delay_ms", int))

# SONIOX_CONTEXT_TERMS=1 이면 Soniox 에 알려 줄 브랜드·업무 용어 (information.json [제품명 리스트] 기준).
# 받아쓰기에 도움이 되는지는 측정으로 확인한 뒤 켠다.
BRAND_TERMS = [
    "하루온", "간 가드 맥스2", "눈가드 맥스3", "눈가드 맥스4", "혈당 가드 맥스2", "조인플렉스", "관절&연골 가드 5days",
    "식물성 멜라토닌 릴렉스 정", "프로폴리스 프로", "콜레스테롤 밸런스", "멀티비타민 올인원", "간 가드 루테잔틴",
    "혈관 순환 세트", "이눌린 배변 가드", "전립선 가드 포 맨", "갱년기 가드 라폰티신", "위 가드 루틴",
    "기관지 가드 브리즈콜", "88한 PICK",
    "주문번호", "송장번호", "무통장입금", "가상계좌", "반품", "교환", "환불", "취소", "상담원",
]


def _soniox_url():
    """main() 의 load_dotenv() 이후, 세션을 열 때마다(=런타임에) 읽어야 한다 — import 시점에
    상수로 굳히면 .env 가 아직 로드되지 않은 값(기본 Soniox 도메인)으로 고정돼 버린다."""
    url = os.environ.get("SONIOX_WS_URL")
    if url:
        return url
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        return base.replace("https://", "wss://", 1).rstrip("/") + "/soniox/transcribe-websocket"
    return "wss://stt-rt.soniox.com/transcribe-websocket"


class SonioxSession:
    """오디오 조각을 넣으면(feed) 받아쓴 텍스트가 쌓이고, 말이 끝나면 end_event 가 켜진다.

    Soniox 가 오류를 보내거나 연결이 끊기면 end_event 가 아니라 failed 를 켠다.
    (말 끝으로 착각하면 손님이 말하는 도중에 답해 그 문항 속도가 0점이 된다.)
    그때는 server.forward_audio 의 음량 안전장치가 말 끝을 판단한다.
    """

    def __init__(self, api_key=None, language_hints=("ko",)):
        self.api_key = api_key or os.environ.get("SONIOX_API_KEY") or os.environ["OPENAI_API_KEY"]
        self.model = os.environ.get("SONIOX_MODEL", "stt-rt-v6")
        self.language_hints = list(language_hints)
        self.ws = None
        self.final_text = ""
        self._partial_tail = ""
        self.end_event = asyncio.Event()
        self.failed = False
        self.error = None
        self._reader_task = None

    def _fail(self, why):
        if not self.failed:
            print(f"[stt] Soniox 사용 불가 — 음량 기준으로 말 끝 판단: {why}", file=sys.stderr)
        self.failed = True
        self.error = why

    def _config(self):
        cfg = {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": RATE,
            "num_channels": 1,
            "language_hints": self.language_hints,
            "enable_endpoint_detection": True,
        }
        for env, key, cast in ENDPOINT_ENV:
            if os.environ.get(env):
                cfg[key] = cast(os.environ[env])
        if os.environ.get("SONIOX_CONTEXT_TERMS") == "1":
            cfg["context"] = {"terms": BRAND_TERMS}
        return cfg

    async def connect(self):
        try:
            self.ws = await websockets.connect(
                _soniox_url(), additional_headers={"Authorization": f"Bearer {self.api_key}"}
            )
            await self.ws.send(json.dumps(self._config()))
        except Exception as e:
            self._fail(f"연결 실패 {e!r}")
            return
        self._reader_task = asyncio.create_task(self._reader())

    async def feed(self, pcm_chunk: bytes):
        if self.ws is not None and not self.end_event.is_set() and not self.failed:
            try:
                await self.ws.send(pcm_chunk)
            except Exception as e:      # 여기서 예외가 나면 forward_audio 태스크(음량 안전장치 포함)가 죽는다
                self._fail(f"전송 실패 {e!r}")

    async def _reader(self):
        try:
            async for msg in self.ws:
                res = json.loads(msg)
                if "error_code" in res or "error_message" in res:
                    self._fail(f"오류 응답 {res}")      # 말 끝이 아니다 — end_event 를 켜면 말 도중에 답하게 된다
                    return
                partial = []
                for t in res.get("tokens", []):
                    if t.get("text") == "<end>":
                        self.end_event.set()
                    elif t.get("is_final"):
                        self.final_text += t["text"]
                    else:
                        partial.append(t.get("text", ""))   # 미확정 토큰은 응답마다 통째로 바뀐다
                self._partial_tail = "".join(partial)
                if res.get("finished"):
                    self.end_event.set()
                    return
        except Exception as e:                       # 연결 끊김·응답 해석 오류 등 모든 예외
            if not self.end_event.is_set():
                self._fail(f"reader 오류 {e!r}")
            return
        if not self.end_event.is_set():
            self._fail("말 끝 없이 스트림 종료")

    @property
    def text(self):
        """지금까지 확정된 텍스트. 말 끝 판정이 늦을 때를 대비해 partial 꼬리까지 합쳐 쓸 수도 있다."""
        return (self.final_text + self._partial_tail).strip()

    async def close(self):
        try:
            if self.ws is not None:
                await self.ws.send("")  # 빈 텍스트 프레임 = 스트림 종료
        except Exception:
            pass
        if self._reader_task:
            try:
                await asyncio.wait_for(self._reader_task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                self._reader_task.cancel()
        if self.ws is not None:
            await self.ws.close()
