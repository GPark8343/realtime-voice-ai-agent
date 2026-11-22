"""OpenAI GPT-5.4 mini 호출 + function calling 오케스트레이션.

LLM이 함수 호출을 요청하면 call_tool(name, args) 로 실제 호출(모의 함수 서버, tools_url 경유)을 실행하고
결과를 다시 LLM에 넘겨 최종 답변 텍스트를 만든다. 한 턴에 함수 호출이 여러 번 이어질 수 있어 상한을 둔다.
"""
import json
import os
import re

from openai import AsyncOpenAI

MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")
MAX_TOOL_ROUNDS = 4

# BAZE 뒤의 모델이 손님(user 역할) 발화 속 전화번호처럼 보이는 숫자열을 함수 인수에 넣을 때
# "[PHONE]" 같은 플레이스홀더로 바꿔치기해 버리는 현상을 실측으로 확인했다(같은 번호라도
# system 메시지로 준 발신자 정보에서는 안 그런다 — user 발화 속 숫자에서만 발생).
# BAZE 공식 문서에도 조직의 개인정보 정책이 Gateway 입력을 마스킹할 수 있다고 적혀 있다
# (https://docs.mindlogic.ai/docs/general/baze/product/api-access) — 팀원(yujin)이 확인.
# 그래서 user 메시지에 넣기 전에 긴 숫자열을 <NUM1>, <NUM2> 토큰으로 바꾸고, 함수 호출 인수에서
# 그 토큰을 다시 원래 숫자로 되돌린다. 모델은 토큰을 그대로 베끼기만 하면 되므로 바꿔치기가 없다.
_NUM_CANDIDATE = re.compile(r"\d[\d\s\-]*\d")


def mask_digit_runs(text, min_digits=7):
    mapping = {}

    def repl(m):
        s = m.group(0)
        if len(re.sub(r"\D", "", s)) < min_digits:
            return s
        token = f"<NUM{len(mapping) + 1}>"
        mapping[token] = s
        return token

    return _NUM_CANDIDATE.sub(repl, text), mapping


def unmask_value(value, mapping):
    if not mapping:
        return value
    if isinstance(value, str):
        for token, original in mapping.items():
            value = value.replace(token, original)
        return value
    if isinstance(value, dict):
        return {k: unmask_value(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [unmask_value(v, mapping) for v in value]
    return value

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                               base_url=os.environ.get("OPENAI_BASE_URL") or None)
    return _client


def summarize_known_facts(history):
    """history 속 tool_call/tool_result 쌍을 뽑아 "이미 확인된 사실" 요약을 만든다.

    history 는 자유 문장으로 섞여 있어, 과거에 이미 호출한 함수와 그 결과를 LLM이 매번 긴 글 속에서
    다시 찾아 추론해야 한다 — 특히 몇 턴 전에 콜백을 이미 접수했거나 이미 조회를 끝낸 경우를
    놓치기 쉽다(실측으로 확인: "이미 접수된 콜백을 모르고 다시 접수/재질문"하는 사례들).
    그래서 같은 정보를 별도의 명확한 목록으로도 만들어 LLM이 추론 없이 바로 참고하게 한다.
    기존 _history_to_messages 의 서술형 메시지는 그대로 두고, 이건 보강용으로 추가한다.
    """
    facts = []
    pending = None
    for turn in history or []:
        role = turn.get("role")
        if role == "tool_call":
            pending = turn
        elif role == "tool_result":
            name = turn.get("name") or (pending or {}).get("name")
            args = (pending or {}).get("arguments") if pending else None
            result = turn.get("result")
            status = (result or {}).get("status") if isinstance(result, dict) else None
            args_txt = json.dumps(args, ensure_ascii=False) if args is not None else "?"
            result_txt = json.dumps(result, ensure_ascii=False)
            if len(result_txt) > 300:
                result_txt = result_txt[:300] + "...(생략)"
            facts.append(f"- {name}({args_txt}) -> status={status}: {result_txt}")
            pending = None
    if not facts:
        return ""
    return (
        "[이번 통화에서 이미 확인된 사실]\n" + "\n".join(facts) +
        "\n위 조회는 다시 부를 필요 없고, 위 접수(취소·반품·교환·콜백)는 다시 접수하지 않습니다. "
        "손님이 그 결과를 다시 묻거나 지적하면, 다시 조회/접수하지 말고 위 내용 그대로 안내합니다."
    )


def _history_to_messages(history):
    """session.start 의 history(글로 된 대화 이력)를 OpenAI 메시지로 변환."""
    msgs = []
    for turn in history or []:
        role = turn.get("role")
        if role == "customer":
            msgs.append({"role": "user", "content": turn.get("text", "")})
        elif role == "agent":
            msgs.append({"role": "assistant", "content": turn.get("text", "")})
        elif role == "tool_call":
            args = json.dumps(turn.get("arguments"), ensure_ascii=False)
            msgs.append({"role": "assistant", "content": f"[이전 함수 호출: {turn.get('name')}({args})]"})
        elif role == "tool_result":
            res = json.dumps(turn.get("result"), ensure_ascii=False)
            msgs.append({"role": "assistant", "content": f"[이전 함수 결과: {turn.get('name')} -> {res}]"})
    return msgs


def build_messages(system_prompt, history, caller, now, customer_utterance):
    """돌려주는 값: (messages, mapping). mapping 은 run_tool_rounds 에 그대로 넘겨 함수 인수를
    원래 숫자로 되돌리는 데 쓴다."""
    context = f"현재 시각: {now}\n"
    if caller:
        context += (
            f"발신자 정보(통신사 발신번호로 확인된 값, 신뢰 가능): {json.dumps(caller, ensure_ascii=False)}\n"
            "손님이 이번 말에서 전화번호나 주문번호를 불렀더라도, 음성 인식 과정에서 숫자가 틀리거나 "
            "깨졌을 수 있습니다. 손님이 부른 번호가 불확실하거나 조회에 실패하면, 다시 묻기 전에 "
            "먼저 이 발신자 전화번호로 조회를 시도하세요."
        )
    else:
        context += "발신자 정보: 확인되지 않음(모름)"

    masked_utterance, mapping = mask_digit_runs(customer_utterance)
    if mapping:
        context += (
            "\n손님 발화 속 긴 숫자열은 <NUM1>, <NUM2> ... 토큰으로 바뀌어 있습니다. 함수 인수에 "
            "그 번호를 넣을 때는 해당 토큰을 그대로(예: \"<NUM1>\") 적으세요. 절대 다른 글자나 "
            "플레이스홀더로 바꿔 쓰지 마세요 — 서버가 토큰을 실제 번호로 되돌립니다."
        )

    facts = summarize_known_facts(history)
    system_messages = [{"role": "system", "content": system_prompt}, {"role": "system", "content": context}]
    if facts:
        system_messages.append({"role": "system", "content": facts})

    messages = (
        system_messages
        + _history_to_messages(history)
        + [{"role": "user", "content": masked_utterance}]
    )
    return messages, mapping


async def first_call(messages, tools):
    """첫 LLM 호출 — 함수 호출 여부를 가장 먼저 알아야 채움말 재생 여부를 바로 결정할 수 있다."""
    client = _get_client()
    resp = await client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, tool_choice="auto", temperature=0.3
    )
    return resp.choices[0].message


async def run_tool_rounds(messages, tools, first_message, call_tool, mapping=None):
    """첫 호출 결과(first_message)부터 시작해, 함수 호출이 끝날 때까지 반복하고 최종 답변 텍스트를 돌려준다.

    call_tool(name, args) -> awaitable[dict] 이어야 한다. mapping 이 있으면 실제로 함수를 부르기
    전에 <NUM..> 토큰을 원래 숫자로 되돌린다(모델에게는 토큰이 담긴 인수를 그대로 둔다).
    돌려주는 값: (answer_text, made_calls: list[{"name","arguments"}]) — arguments 는 실제 호출에
    쓴(토큰을 되돌린) 값이다.
    """
    client = _get_client()
    msg = first_message
    made_calls = []
    for _ in range(MAX_TOOL_ROUNDS):
        if not msg.tool_calls:
            return msg.content or "", made_calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            real_args = unmask_value(args, mapping)
            result = await call_tool(name, real_args)
            made_calls.append({"name": name, "arguments": real_args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})
        resp = await client.chat.completions.create(model=MODEL, messages=messages, tools=tools,
                                                      tool_choice="auto", temperature=0.3)
        msg = resp.choices[0].message
    return msg.content or "", made_calls
