"""TTS — 행사 지원 LLM 게이트웨이(BAZE, OpenAI 호환 REST)를 통해 Gemini TTS로 답변을 합성한다.

주의: 스펙에 적힌 모델명은 `gemini-3.8-flash-lite-tts` 이지만, 이 게이트웨이에는 그 이름이 없고
`gemini-3.8-flash-tts` 만 있다(404 확인됨). 그래서 기본값을 `gemini-3.8-flash-tts` 로 둔다.
실측 결과 이 백엔드는 진짜 점진적 스트리밍이 아니라 통째로 합성한 뒤 한 번에 내려준다
(문구 하나에 TTFB 약 2.5~3초). 그래서 서버는 미리 만들어 둔 맞장구·대기 멘트(cues/*.wav)를 먼저
내보내는 방식으로 첫 소리 지연을 줄인다.
"""
import os

import numpy as np
from openai import AsyncOpenAI
from scipy.signal import resample_poly

MODEL = os.environ.get("TTS_MODEL", "gemini-3.8-flash-tts")
VOICE = os.environ.get("TTS_VOICE", "Kore")
OUT_RATE = 16000
SRC_RATE = int(os.environ.get("TTS_SRC_RATE", "24000"))  # 실측으로 확정되기 전까지의 가정치

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                               base_url=os.environ.get("OPENAI_BASE_URL") or None)
    return _client


def _resample(pcm_bytes: bytes) -> bytes:
    if not pcm_bytes:
        return b""
    if SRC_RATE == OUT_RATE:
        return pcm_bytes
    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(a) == 0:
        return b""
    out = resample_poly(a, OUT_RATE, SRC_RATE)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


async def synthesize(text: str) -> bytes:
    """답변 텍스트 -> 16kHz/16bit/mono PCM.

    받은 조각을 모두 모은 뒤 한 번에 리샘플한다. 조각마다 따로 리샘플하면 조각 경계에 딸깍 소리가 나고,
    조각이 홀수 바이트로 끊겨 오면 변환 자체가 실패한다. 이 백엔드는 어차피 통째로 한 번에 내려주므로
    모아서 변환해도 첫 소리가 늦어지지 않는다.
    """
    text = (text or "").strip()
    if not text:
        return b""
    async with _get_client().audio.speech.with_streaming_response.create(
        model=MODEL, voice=VOICE, input=text, response_format="pcm"
    ) as resp:
        raw = b"".join([c async for c in resp.iter_bytes()])
    return _resample(raw[: len(raw) // 2 * 2])  # 끝에 남는 홀수 바이트는 버린다(16bit 샘플이 아님)
