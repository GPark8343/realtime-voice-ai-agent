"""실시간 음성 AI 상담원 — WebSocket 서버 (STT → LLM(Function Calling) → TTS).

SPEC.md 의 통신 규격을 따른다. 문항 하나 = 연결 하나 = 손님 말 한 번에 대한 응대 한 번.

흐름:
  1. session.start 수신
  2. 손님 음성을 Soniox 실시간 STT 로 흘려 넣으며 말 끝을 기다린다
     (Soniox 의 <end> 토큰이 1차 기준, 받아쓰기 쪽에서 신호가 없을 때를 대비해 RMS 무음 타임아웃을 보조로 둔다)
  3. 말 끝을 판정하면 바로, 미리 만들어 둔 맞장구("네, 고객님." 등)를 보내 첫 소리를 낸다(cues.py).
     동시에 LLM(GPT-5.4 mini) 을 호출하고, 함수 호출이 필요하면 그 함수에 맞는 대기 멘트를 먼저 보낸다.
  4. 함수 호출이 필요하면 tools_url(측정기 중계) 로 실행하고 결과를 반영해 최종 답변을 만든다.
  5. 답변을 TTS(Gemini) 로 합성하며 조각이 나오는 대로 바로 전송한다.
  6. turn.end 를 보낸다.

실행: python3 server.py --port 9000
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

import httpx
import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()  # 아래 모듈들이 import 시점에 환경변수(OPENAI_BASE_URL 등)를 읽으므로 먼저 로드한다.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
import cues as cuesmod
import prompt as promptmod
import stt as sttmod
import tts as ttsmod

RATE = 16000
SPEECH_RMS = 500             # 이보다 크면 말소리로 본다
FALLBACK_SILENCE_MS = 2500   # Soniox <end> 가 안 올 때 대비한 보조 무음 타임아웃
FAILED_SILENCE_MS = 1500     # Soniox 가 죽었을 때(연결 실패·오류) 무음 기준. 현장 설명: 한 발화 안에서 1.5초 이상 쉬는 손님은 없다
MAX_WAIT_S = 20.0            # 말 끝을 기다리는 전체 상한(측정기 25초 제한보다 여유를 둔다)
# 말 끝 판정 뒤 맞장구까지 기다릴 시간. 0 이면 바로 보낸다. 그동안 손님 소리가 다시 들리면 맞장구를 건너뛴다.
# Soniox 가 말 끝을 너무 일찍 판정하는 비율을 측정한 뒤 조정한다(말 도중 첫 소리 = 그 문항 속도 0점).
ACK_DELAY_MS = int(os.environ.get("ACK_DELAY_MS", "0"))

SYSTEM_PROMPT = None
TOOLS = None
CUE_BANK = cuesmod.CueBank({})   # main() 에서 cues/*.wav 를 읽어 채운다


def rms(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    """문장 단위로 쪼갠다. TTS 한 번 호출에 고정 지연(~2.5~3초)이 있고 길이에 비례해 더 늘어나므로,
    문장마다 따로 합성해 먼저 끝난 문장부터 보내면 '조회 뒤 답' 첫 소리 지연을 크게 줄일 수 있다."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


async def speak(ws, text):
    """문장별로 TTS 를 동시에(파이프라인으로) 돌리고, 순서대로 끝나는 대로 바로 전송한다."""
    sentences = split_sentences(text)
    tasks = [asyncio.create_task(ttsmod.synthesize(s)) for s in sentences]
    for t in tasks:
        pcm = await t
        if pcm:
            await ws.send(pcm)


class VoiceTracker:
    """마지막으로 손님 소리(RMS ≥ SPEECH_RMS)가 들린 시각. 맞장구 직전에 손님이 다시 말하는지 볼 때 쓴다."""

    def __init__(self):
        self.last_t = 0.0


async def forward_audio(ws, stt_session, fallback_event, voice=None):
    """손님 음성 프레임을 STT 로 전달하고, RMS 로 보조 무음 타임아웃을 감시한다.

    측정기는 말이 끝난 뒤에도 계속 음성을 보내므로, 이 태스크는 연결이 끝날 때까지 계속 돈다.
    voice 를 주면 손님 소리가 들릴 때마다 그 시각을 기록한다.
    """
    heard, silent_ms = False, 0
    async for m in ws:
        if not isinstance(m, bytes):
            continue
        if not fallback_event.is_set():
            await stt_session.feed(m)
        r = rms(m)
        if r >= SPEECH_RMS:
            heard, silent_ms = True, 0
            if voice is not None:
                voice.last_t = time.monotonic()
        elif heard:
            silent_ms += len(m) * 1000 // (RATE * 2)
            limit = FAILED_SILENCE_MS if getattr(stt_session, "failed", False) else FALLBACK_SILENCE_MS
            if silent_ms >= limit:
                fallback_event.set()


async def send_ack(ws, eou_t, voice, turn=0):
    """말 끝 판정 직후 맞장구를 보낸다(첫 소리). turn(상담원 발화 수)으로 맞장구를 번갈아 고른다.
    ACK_DELAY_MS 동안 손님 소리가 다시 들리면 건너뛴다."""
    pcm = CUE_BANK.ack(turn)
    if not pcm:
        return
    if ACK_DELAY_MS > 0:
        await asyncio.sleep(ACK_DELAY_MS / 1000)
        if voice.last_t > eou_t:
            return
    await ws.send(pcm)


async def wait_for_speech_end(stt_session, fallback_event):
    tasks = [asyncio.create_task(stt_session.end_event.wait()),
             asyncio.create_task(fallback_event.wait())]
    try:
        await asyncio.wait_for(
            asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED), timeout=MAX_WAIT_S
        )
    except asyncio.TimeoutError:
        pass
    for t in tasks:
        t.cancel()


def tool_args(call, mapping):
    """LLM 이 정한 함수 인자(JSON 문자열) → dict. <NUM..> 토큰은 원래 숫자로 되돌려 history 의 인자와 비교할 수 있게 한다.
    대기 멘트를 고르는 데만 쓰므로, JSON 이 깨졌으면 None 을 돌려주고 응대는 그대로 계속한다."""
    try:
        return llm.unmask_value(json.loads(call.function.arguments or "{}"), mapping)
    except (json.JSONDecodeError, TypeError):
        return None


def make_call_tool(http, tools_url, session_id):
    async def call_tool(name, args):
        try:
            r = await http.post(f"{tools_url.rstrip('/')}/tools/{name}", json=args,
                                 headers={"X-Session-Id": session_id}, timeout=30)
            return r.json()
        except Exception as e:
            return {"status": "error", "message": f"함수 호출 실패: {e!r}"[:200]}
    return call_tool


async def handle_turn(ws, http, start):
    stt_session = sttmod.SonioxSession()
    await stt_session.connect()
    fallback_event = asyncio.Event()
    voice = VoiceTracker()
    forward_task = asyncio.create_task(forward_audio(ws, stt_session, fallback_event, voice))
    ack_task = None

    try:
        await wait_for_speech_end(stt_session, fallback_event)
        # 말 끝 판정 즉시 맞장구 = 첫 소리. LLM 호출이 늦어지지 않게 별도 태스크로 띄운다.
        ack_task = asyncio.create_task(
            send_ack(ws, time.monotonic(), voice, cuesmod.agent_turns(start.get("history"))))
        customer_text = stt_session.text
        if os.environ.get("DEBUG_TRANSCRIPT"):
            print(f"[stt] {start.get('session_id')}: {customer_text!r}", file=sys.stderr)
        await stt_session.close()
        if not customer_text:
            # 아무것도 받아쓰지 못했다 — 되물어보는 안내로 대체한다.
            customer_text = "(음성을 알아듣지 못함)"

        messages, mapping = llm.build_messages(SYSTEM_PROMPT, start.get("history"), start.get("caller"),
                                                start.get("now"), customer_text)
        call_tool = make_call_tool(http, start["tools_url"], start["session_id"])

        try:
            msg = await llm.first_call(messages, TOOLS)
        finally:
            # 맞장구가 대기 멘트·본 답보다 먼저 나가도록 순서를 지킨다.
            # LLM 이 실패해도 맞장구는 끝까지 보낸다 — 소리가 하나도 없으면 그 문항 속도가 0점이다.
            await ack_task

        if msg.tool_calls:
            # 대기 멘트 — 부를 함수와 인자를 이미 알고 있으므로 그에 맞는 말을 고른다(직전 조회와 같은 함수면 반복을 피함)
            call = msg.tool_calls[0]
            filler = CUE_BANK.tool_filler(call.function.name, tool_args(call, mapping), start.get("history"))
            if filler:
                await ws.send(filler)
            answer, _calls = await llm.run_tool_rounds(messages, TOOLS, msg, call_tool, mapping)
        else:
            answer = msg.content or ""

        answer = answer.strip() or "죄송합니다, 다시 한 번 말씀해 주시겠어요?"
        if os.environ.get("DEBUG_TRANSCRIPT"):
            await ws.send(json.dumps({"type": "agent.transcript", "text": answer}, ensure_ascii=False))

        await speak(ws, answer)

        await ws.send(json.dumps({"type": "turn.end"}))
    finally:
        if ack_task is not None and not ack_task.done():
            ack_task.cancel()
        forward_task.cancel()
        try:
            await stt_session.close()
        except Exception:
            pass


async def handler(ws, http):
    try:
        start = json.loads(await ws.recv())
        assert start.get("type") == "session.start"
        await handle_turn(ws, http, start)
    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[error] {e!r}", file=sys.stderr)
        try:
            await ws.send(json.dumps({"type": "turn.end"}))
        except Exception:
            pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()

    global SYSTEM_PROMPT, TOOLS, CUE_BANK
    SYSTEM_PROMPT = promptmod.build_system_prompt()
    TOOLS = promptmod.load_tools()
    # 미리 만든 음성 파일만 읽는다 — 켤 때 TTS(크레딧) 호출이 없고, 측정 당일 TTS 장애와도 무관하다
    CUE_BANK = cuesmod.CueBank.load()
    print(f"맞장구·대기 멘트 {len(CUE_BANK)}/{len(cuesmod.CUE_TEXTS)}개 준비", file=sys.stderr)

    async with httpx.AsyncClient() as http:
        async with websockets.serve(lambda ws: handler(ws, http), "0.0.0.0", a.port, max_size=None):
            print(f"agent server on :{a.port}", file=sys.stderr)
            await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
