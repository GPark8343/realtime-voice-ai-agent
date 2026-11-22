"""맞장구·대기 멘트 음성 만들기 — 게이트웨이 TTS 를 불러 cues/<키>.wav 를 만든다. (크레딧 사용!)

대사(cues.CUE_TEXTS)를 바꾸면 이 스크립트를 다시 돌리고 결과 WAV 와 manifest.json 을 커밋한다.
manifest.json 은 각 WAV 를 어떤 대사로 만들었는지 적어 둔 것이다. 대사만 고치고 음성을 다시 안 만들면
tests/test_cues.py::test_manifest_matches_texts 가 실패한다.

실행: python agent_server/make_cues.py              # 전부
      python agent_server/make_cues.py ack_1 order  # 일부만
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # tts 모듈이 import 시점에 TTS_MODEL 등을 읽으므로 먼저 로드한다

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cues
import tts


async def main(keys):
    cues.CUE_DIR.mkdir(exist_ok=True)
    manifest_path = cues.CUE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for key in keys:
        text = cues.CUE_TEXTS[key]
        pcm = cues.trim_silence(await tts.synthesize(text))
        level = cues.rms(pcm)
        if level < cues.MIN_CUE_RMS:
            # 너무 작으면 측정기가 무음으로 보고 무시한다 — 저장하지 않고 알린다(빈 결과도 여기서 걸린다)
            print(f"{key}: 소리가 너무 작아 저장하지 않음 (RMS {level:.0f} < {cues.MIN_CUE_RMS})", file=sys.stderr)
            continue
        cues.write_wav(cues.CUE_DIR / f"{key}.wav", pcm)
        manifest[key] = text
        # 하나 만들 때마다 바로 적는다 — 중간에 멈춰도 디스크의 WAV 와 manifest 가 어긋나지 않게
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{key}: {text}  ({len(pcm) / 2 / cues.RATE:.2f}초, RMS {level:.0f})")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(cues.CUE_TEXTS)
    unknown = [k for k in keys if k not in cues.CUE_TEXTS]
    if unknown:
        sys.exit(f"모르는 키: {unknown} (가능: {list(cues.CUE_TEXTS)})")
    asyncio.run(main(keys))
