"""server.forward_audio 의 음량 안전장치 테스트 — Soniox 가 죽었을 때 무음 기준이 1.5초로 당겨지는지 본다."""
import asyncio

import numpy as np

import server


class FrameWS:
    """측정기 흉내: 정해진 음성 조각을 차례로 내준다."""

    def __init__(self, frames):
        self.frames = frames

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for f in self.frames:
            yield f


class StubSTT:
    def __init__(self, failed):
        self.failed = failed

    async def feed(self, pcm):
        pass


LOUD = np.full(320, 3000, dtype=np.int16).tobytes()   # 20ms 말소리
QUIET = b"\0" * 640                                     # 20ms 무음


def fallback_fired(silence_ms, failed):
    frames = [LOUD] * 10 + [QUIET] * (silence_ms // 20)
    ev = asyncio.Event()
    asyncio.run(server.forward_audio(FrameWS(frames), StubSTT(failed), ev))
    return ev.is_set()


def test_healthy_stt_keeps_existing_fallback():
    assert not fallback_fired(2000, failed=False)
    assert fallback_fired(server.FALLBACK_SILENCE_MS, failed=False)


def test_failed_stt_uses_1_5s_silence():
    """현장 설명: 한 발화 안에서 1.5초 이상 쉬는 손님은 없다 → 1.5초 무음이면 끼어들 위험 없이 말 끝."""
    assert not fallback_fired(1400, failed=True)
    assert fallback_fired(1500, failed=True)
