import datetime
from utils import date_utils, text_utils

def test_parse_date():
    assert date_utils.parse_date("2024-03-08") == datetime.date(2024, 3, 8)
    assert date_utils.parse_date(None) is None
    assert date_utils.parse_date("invalid") is None

def test_months_between():
    start = datetime.date(2024, 1, 1)
    end = datetime.date(2024, 3, 1)
    assert date_utils.months_between(start, end) == 2
    assert date_utils.months_between(start, None) > 20

def test_total_career_months():
    history = [
        {"duration_months": 12},
        {"duration_months": 15}
    ]
    assert date_utils.total_career_months(history) == 27
    assert date_utils.total_career_months([]) == 0

def test_clean_text():
    assert text_utils.clean_text("  Hello World\r\n") == "hello world"
    assert text_utils.clean_text("") == ""

def test_concat_career_text():
    candidate = {
        "profile": {
            "headline": "Lead AI",
            "summary": "Building search engines"
        },
        "career_history": [
            {"title": "SE", "description": "Backend python"}
        ]
    }
    res = text_utils.concat_career_text(candidate)
    assert "Lead AI" in res
    assert "Building search engines" in res
    assert "SE" in res
    assert "Backend python" in res

def test_tokenize_for_bm25():
    text = "Hello, world! Building ML-models..."
    tokens = text_utils.tokenize_for_bm25(text)
    assert tokens == ["hello", "world", "building", "mlmodels"]
