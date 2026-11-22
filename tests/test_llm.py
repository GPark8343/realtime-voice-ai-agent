"""mask_digit_runs/unmask_value 테스트 — PII 마스킹이 서로 다른 번호를 한 토큰으로 섞지 않는지 확인."""
import llm


def test_short_chunks_merge_into_one_phone_number():
    """"010 5182 4529" 처럼 한 자리씩 끊어 부른 전화번호는 그대로 하나로 합친다."""
    masked, mapping = llm.mask_digit_runs("010 5182 4529")
    assert masked == "<NUM1>"
    assert mapping == {"<NUM1>": "010 5182 4529"}


def test_hyphenated_order_id_inside_sentence():
    masked, mapping = llm.mask_digit_runs("주문확인 20260929-0033428 금원서")
    assert masked == "주문확인 <NUM1> 금원서"
    assert mapping == {"<NUM1>": "20260929-0033428"}


def test_short_number_below_min_digits_is_not_masked():
    masked, mapping = llm.mask_digit_runs("456 개")
    assert masked == "456 개"
    assert mapping == {}


def test_unmask_value_restores_tokens_in_nested_args():
    mapping = {"<NUM1>": "2956732601515352", "<NUM2>": "01082389762"}
    args = {"order_id": "<NUM1>", "buyer_cellphone": "<NUM2>"}
    assert llm.unmask_value(args, mapping) == {
        "order_id": "2956732601515352",
        "buyer_cellphone": "01082389762",
    }
