"""모의 함수 서버 — 쇼핑몰 시스템 대신 함수(주문 조회·반품 접수 등)에 응답한다.

공식 함수 정의로 인수를 검증하고 제공된 데이터에서 응답을 고른다.
호출 기록과 응답 지연은 측정기(run_eval.py)의 함수 중계가 맡으므로 여기서 할 필요가 없다.
지켜야 할 규칙은 SPEC.md 의 「모의 함수 서버」에 있다.

실행:  python3 mock_server.py --port 8100 [--host 0.0.0.0] [--db ../data/mock_db.json]

  GET  /tools                 함수 목록 (OpenAI tools 형식, data/brand/tools.json)
  POST /tools/{함수명}         body = 인수 JSON. 헤더 X-Session-Id 에 session.start 로 받은 session_id 가 온다.
  GET  /calls?session_id=...  (참고용) 이 서버가 받은 호출 기록. 채점에는 쓰지 않는다.
"""
import argparse, difflib, json, os, re, sys, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))

# 조회 함수: 들어온 인수와 기록의 인수를 비교할 때 쓰는 식별 필드들.
QUERY_MATCH_KEYS = {
    "get_order_info": ["order_id", "buyer_cellphone"],
    "get_return_info": ["customer_phone_number"],
}

# 접수 함수: 같은 건의 실제 기록을 찾을 때 쓰는 식별 필드. 기록이 없어도 필수 값이 다 있으면 ok 다.
ACCEPT_MATCH_KEYS = {
    "cancel_order": "order_id",
    "return_order": "order_number",
    "exchange_order": "order_number",
    "request_callback": "phone_number",
}


def _norm(v):
    """표기 차이(하이픈·공백·구두점·대소문자)를 무시하기 위한 정규화. 값이 없으면 빈 문자열.
    숫자만 남기면 'absent' 같은 엉뚱한 값이 빈 값으로 바뀌어 '입력 없음'과 구분이 안 되므로 글자는 남긴다."""
    return re.sub(r"[\W_]+", "", str(v)).casefold() if v is not None else ""


def _blank(v):
    """필수 값이 빠진 것으로 볼 값: None, 빈 목록, 공백뿐인 문자열."""
    return v is None or v == [] or (isinstance(v, str) and not v.strip())


class Store:
    def __init__(self, db_path, tools_path):
        # DB 파일이 없으면 조용히 빈 DB 로 뜨지 않고 바로 실패한다 (경로 실수를 측정 전에 알 수 있게).
        self.db = json.loads(Path(db_path).read_text(encoding="utf-8"))   # {함수명: [{"arguments", "result"}]}
        self.tools = json.loads(Path(tools_path).read_text(encoding="utf-8"))
        self.required = {t["name"]: t.get("parameters", {}).get("required", []) for t in self.tools}
        from jsonschema import validators
        self.validators = {}
        for tool in self.tools:
            validator = validators.validator_for(tool["parameters"])
            validator.check_schema(tool["parameters"])
            self.validators[tool["name"]] = validator(tool["parameters"])
        self.calls = []
        self.lock = threading.Lock()

    def _missing_fields(self, name, args):
        """빠진 필수 값 이름 목록. 빈 문자열·공백·빈 목록도 빠진 것으로 본다."""
        return [f for f in self.required.get(name, []) if _blank(args.get(f))]

    @staticmethod
    def _bad_return_items(args):
        """return_order_item 의 상품 목록 세부 확인 (형식 검사를 통과한 뒤에 부른다)."""
        return any(_blank(it["order_item_code"]) or it["quantity"] <= 0 for it in args["return_items"])

    def _query_match(self, name, args):
        supplied = [k for k in QUERY_MATCH_KEYS[name] if _norm(args.get(k))]
        best, best_hits = None, 0
        # 공식 채점 기준(SCORING.md)은 전화번호 또는 주문번호 "중 하나"만 정답과 같으면 맞은 호출로
        # 본다 — 같이 들어온 다른 필드가 음성 인식 오류로 틀려도(예: 깨진 주문번호 + 맞는 전화번호)
        # 맞는 필드 하나로 매칭한다. 다른 필드가 틀렸다고 전체를 거부하지 않는다. (PR #4)
        # 같은 주문을 여러 번 조회한 기록은 시간순으로 쌓여 있다 → 뒤(최신)부터 본다.
        for rec in reversed(self.db.get(name, [])):
            rargs = rec["arguments"]
            hits = sum(1 for k in supplied if _norm(rargs.get(k)) == _norm(args[k]))
            # 더 많이 맞는 기록(주문번호+전화번호)이 하나만 맞는 기록보다 우선.
            # 같은 개수면 먼저 본 것(최신)을 유지한다.
            if hits > best_hits:
                best, best_hits = rec["result"], hits
        return best

    def _order_in_results(self, args):
        """주문번호가 기록의 인수엔 없고 전화번호로 조회한 결과 안에만 있는 경우 (연습 DB 에 20건).
        그 주문만 골라 실제 주문번호 조회처럼 돌려준다."""
        target = _norm(args.get("order_id"))
        if not target:
            return None
        for rec in reversed(self.db.get("get_order_info", [])):
            res = rec["result"]
            orders = (res.get("data") or {}).get("orders") or []
            hit = [o for o in orders if _norm(o.get("주문번호")) == target]
            if hit:
                return {**res, "data": {**res["data"], "orders": hit}}
        return None

    def _accept_match(self, name, args):
        key = ACCEPT_MATCH_KEYS.get(name)
        if key:
            target = _norm(args.get(key))
            for rec in reversed(self.db.get(name, [])):  # 조회와 같이 최신 기록 우선
                if _norm(rec["arguments"].get(key)) == target:
                    return rec["result"]
        elif name == "return_order_item":
            codes = {_norm(it.get("order_item_code")) for it in args.get("return_items", [])}
            for rec in reversed(self.db.get(name, [])):
                rec_codes = {_norm(it.get("order_item_code")) for it in rec["arguments"].get("return_items", [])}
                if codes & rec_codes:
                    return rec["result"]
        return None

    def _search_products(self, args):
        records = self.db.get("search_products", [])
        if not records:
            return {"status": "not_found"}, False
        query = _norm(args["query"])
        best = max(records, key=lambda r: difflib.SequenceMatcher(None, query, _norm(r["arguments"].get("query"))).ratio())
        return best["result"], True

    def lookup(self, name, args):
        """(응답 JSON, 기록과 맞았는지 bool) 을 돌려준다. SPEC.md 「모의 함수 서버」 규칙을 따른다."""
        from jsonschema import ValidationError

        if name not in self.validators:
            return {"status": "not_found"}, False
        if not isinstance(args, dict):
            return {"status": "invalid"}, False
        # 1) 필수 값 검사: 무엇이 빠졌는지 알려 주면 상담 AI 가 손님에게 그 값을 다시 물을 수 있다
        missing = self._missing_fields(name, args)
        if missing:
            return {"status": "invalid", "message": f"다음 값이 필요합니다: {', '.join(missing)}"}, False
        # 2) 함수 정의(JSON Schema)로 형식 검사: 타입이 틀리면 invalid
        try:
            self.validators[name].validate(args)
        except ValidationError:
            return {"status": "invalid", "message": "인수 형식이 함수 정의와 맞지 않습니다"}, False
        if name == "return_order_item" and self._bad_return_items(args):
            return {"status": "invalid", "message": "다음 값이 필요합니다: return_items"}, False

        # 3) 조회 함수: 전화번호·주문번호가 맞는 기록의 result (get_order_info 는 필수 값이 없어 둘 중 하나는 있어야 함)
        if name in QUERY_MATCH_KEYS:
            if not any(_norm(args.get(k)) for k in QUERY_MATCH_KEYS[name]):
                return {"status": "invalid", "message": "주문번호나 전화번호가 필요합니다"}, False
            result = self._query_match(name, args)
            if result is None and name == "get_order_info":
                result = self._order_in_results(args)
            if result is not None:
                return result, True
            return {"status": "not_found"}, False

        # 4) 검색 함수: 가장 가까운 검색어의 기록
        if name == "search_products":
            return self._search_products(args)

        # 5) 접수 함수: 같은 건의 실제 기록이 있으면 그 결과(예: already_requested)를 그대로 돌려준다.
        #    채점 AI 는 실제 상담 때의 함수 결과를 정답 참고로 보므로(eval/score.py judge()),
        #    기록을 재현해야 상담 AI 가 "이미 접수돼 있습니다"처럼 실제 상황에 맞게 답할 수 있다.
        #    기록이 없으면 SPEC 대로 필수 값이 다 있으니 ok.
        result = self._accept_match(name, args)
        if result is not None:
            return result, True
        return {"status": "ok"}, False


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/tools":
                return self._send(200, [{"type": "function", "function": t} for t in store.tools])
            if u.path == "/calls":
                cid = parse_qs(u.query).get("session_id", [None])[0]
                with store.lock:
                    return self._send(200, [c for c in store.calls if cid is None or c["session_id"] == cid])
            self._send(404, {"error": "not found"})

        def do_POST(self):
            m = re.fullmatch(r"/tools/(\w+)", urlparse(self.path).path)
            if not m or m.group(1) not in {t["name"] for t in store.tools}:
                return self._send(404, {"error": "unknown function"})
            name = m.group(1)
            t0 = time.time()
            try:
                args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return self._send(400, {"error": "body must be JSON"})
            result, matched = store.lookup(name, args)
            # 참고용 호출 기록 (디버깅용)
            rec = {"session_id": self.headers.get("X-Session-Id"), "name": name, "arguments": args, "matched": matched,
                   "t_request": t0, "t_response": time.time()}
            with store.lock:
                store.calls.append(rec)
            self._send(200, result)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--db", default=os.path.join(HERE, "..", "data", "mock_db.json"))
    ap.add_argument("--tools", default=os.path.join(HERE, "..", "data", "brand", "tools.json"))
    a = ap.parse_args()
    store = Store(a.db, a.tools)
    print(f"mock tools on :{a.port}  functions={len(store.tools)}  records={sum(map(len, store.db.values()))}",
          file=sys.stderr)
    ThreadingHTTPServer((a.host, a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
