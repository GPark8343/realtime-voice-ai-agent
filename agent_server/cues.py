"""맞장구·대기 멘트 — 미리 합성해 둔 짧은 음성(cues/*.wav)을 읽어 두었다가 바로 꺼내 쓴다.

왜 미리 만들어 두나: 게이트웨이 TTS 는 한 번 부를 때 2.5~3초가 걸려서, 말 끝 직후 1.5초 안에 첫 소리를
내려면 이미 만들어 둔 소리를 보내는 수밖에 없다. 서버를 켤 때 합성하지 않고 파일로 커밋해 두면
켤 때 크레딧이 들지 않고, 측정 당일 TTS 장애와도 무관하다. 음성은 make_cues.py 로 만든다.
"""
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16000
CUE_DIR = Path(__file__).resolve().parent / "cues"

# 키 → 대사 (사용자 검토 완료, 설계 문서 3장). 대사를 바꾸면 make_cues.py 로 음성을 다시 만든다.
CUE_TEXTS = {
    "ack_1": "네, 고객님.",
    "ack_2": "네, 잠시만요.",
    "order": "주문 내역 확인해 볼게요.",
    "return_info": "반품 내역 확인해 볼게요.",
    "search": "상품 정보 찾아볼게요.",
    "cancel": "취소 접수 도와드릴게요.",
    "return": "반품 접수 도와드릴게요.",
    "exchange": "교환 접수 도와드릴게요.",
    "callback": "상담원이 연락드리도록 요청해 둘게요.",
    # 멀티턴: 가장 최근 함수 호출과 같은 함수를 또 부를 때 함수 이름 대사 대신 쓴다(같은 말 반복 방지)
    "follow_up": "말씀하신 건으로 확인해 볼게요.",
    "recheck": "다시 확인해 볼게요.",
}
ACK_KEYS = ("ack_1", "ack_2")

# 함수 이름 → 대기 멘트 키. 대기 멘트가 나갈 때는 LLM 이 부를 함수를 이미 알고 있으므로 그에 맞는 말을 고른다.
TOOL_CUE = {
    "get_order_info": "order",
    "get_return_info": "return_info",
    "search_products": "search",
    "cancel_order": "cancel",
    "return_order": "return",
    "return_order_item": "return",
    "exchange_order": "exchange",
    "request_callback": "callback",
}

SILENCE_RMS = 300   # 측정기(eval/run_eval.py)가 무음으로 보는 기준과 같게 둔다
WINDOW_MS = 10
PAD_MS = 30
# 맞장구는 한 조각으로 보내고, 측정기는 조각 전체 RMS 가 SILENCE_RMS 이상이어야 소리로 센다.
# 조금 작게 합성돼도 안전하도록 두 배 여유를 둔다(make_cues.py 와 테스트가 이 값으로 확인한다).
MIN_CUE_RMS = 2 * SILENCE_RMS


def rms(pcm: bytes) -> float:
    """조각 전체의 RMS — 측정기(eval/run_eval.py)와 같은 계산. 빈 조각은 0."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt((a * a).mean())) if len(a) else 0.0


def trim_silence(pcm: bytes) -> bytes:
    """앞뒤 무음을 잘라 낸다(앞뒤 PAD_MS 만 남김). 전부 무음이면 b"".

    측정기는 받은 조각 하나 '전체'의 RMS 가 300 이상이면 그 조각이 도착한 시각을 첫 소리로 기록한다.
    맞장구는 한 조각으로 보내므로, 앞뒤 무음을 잘라 내면 (1) 조각 전체 RMS 가 올라가 소리로 확실히 잡히고
    (2) 사람이 듣는 첫 소리도 그만큼 빨라진다.
    10ms 창마다 RMS 를 재서 300 이상인 첫 창과 마지막 창 사이만 남긴다.
    """
    a = np.frombuffer(pcm, dtype=np.int16)
    win = RATE * WINDOW_MS // 1000
    n = len(a) // win
    if n == 0:
        return b""
    frames = a[: n * win].astype(np.float32).reshape(n, win)
    loud = np.flatnonzero(np.sqrt((frames ** 2).mean(axis=1)) >= SILENCE_RMS)
    if len(loud) == 0:
        return b""
    pad = RATE * PAD_MS // 1000
    start = max(0, loud[0] * win - pad)
    end = min(len(a), (loud[-1] + 1) * win + pad)
    return a[start:end].tobytes()


def read_wav(path: Path) -> bytes:
    """16kHz/16bit/mono WAV 의 PCM 을 돌려준다. 형식이 다르면 ValueError (측정기 규격과 맞아야 하므로)."""
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
            raise ValueError(f"{Path(path).name}: 16kHz/16bit/mono 가 아님")
        pcm = w.readframes(w.getnframes())
        if len(pcm) != w.getnframes() * 2:
            raise ValueError(f"{Path(path).name}: 데이터가 잘림")
        return pcm


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def last_tool_call(history):
    """history 를 뒤에서부터 훑어 가장 최근 함수 호출 기록을 찾는다. 없으면 None.

    마지막 기록은 보통 상담원 말("어느 주문이세요?")이므로 마지막 한 줄만 보면 안 된다.
    """
    for h in reversed(history or []):
        if h.get("role") == "tool_call":
            return h
    return None


def agent_turns(history):
    """history 속 상담원 발화 수 — 맞장구 순번으로 쓴다(서버는 이전 턴을 기억하지 않으므로)."""
    return sum(1 for h in history or [] if h.get("role") == "agent")


class CueBank:
    """키 → PCM 을 들고 있다가 맞장구·대기 멘트를 꺼내 준다. 없는 소리는 b"" (호출부는 보내지 않는다)."""

    def __init__(self, pcm_by_key):
        self._pcm = {k: v for k, v in pcm_by_key.items() if v}

    @classmethod
    def load(cls, cue_dir=CUE_DIR):
        """cue_dir/<키>.wav 를 읽는다. 없거나 깨진 파일은 경고만 남기고 건너뛴다(서버는 반드시 떠야 하므로)."""
        pcm = {}
        for key in CUE_TEXTS:
            path = Path(cue_dir) / f"{key}.wav"
            try:
                pcm[key] = read_wav(path)
            except (OSError, wave.Error, ValueError, EOFError) as e:
                print(f"[cues] {key} 음성을 읽지 못해 건너뜀: {e}", file=sys.stderr)
        return cls(pcm)

    def __len__(self):
        return len(self._pcm)

    def ack(self, turn=0):
        """맞장구를 순번(turn)으로 번갈아 고른다. 무작위면 연속한 두 턴에 같은 말이 절반쯤 나오지만,
        순번이면 연속 반복이 없다. 같은 말만 반복하면 사용자 경험 감점 사례다."""
        keys = [k for k in ACK_KEYS if k in self._pcm]
        return self._pcm[keys[turn % len(keys)]] if keys else b""

    def tool_filler(self, tool_name, args=None, history=None):
        """함수에 맞는 대기 멘트. 모르는 함수거나 음성이 없으면 b"".

        history 의 가장 최근 함수 호출이 같은 함수면, 인자가 다를 때(손님이 알려 준 번호로 새 조회)
        "말씀하신 건으로 확인해 볼게요.", 인자까지 같을 때(진짜 재조회) "다시 확인해 볼게요."를 쓴다.
        """
        key = TOOL_CUE.get(tool_name, "")
        prev = last_tool_call(history)
        if key and prev is not None and prev.get("name") == tool_name:
            key = "recheck" if prev.get("arguments") == args else "follow_up"
        return self._pcm.get(key, b"")
