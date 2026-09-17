from __future__ import annotations

import pytest

from privaudit.data.schema import build_schema


def test_schema_with_text_template():
    schema = build_schema({
        "format": "csv",
        "member_file": "train.csv",
        "nonmember_file": "test.csv",
        "text_template": "Q: {question}\nA: {answer}",
    })
    assert schema.row_to_text({"question": "Dose?", "answer": "10mg"}) == "Q: Dose?\nA: 10mg"


def test_schema_with_text_template_supports_nested_access():
    schema = build_schema({
        "format": "jsonl",
        "member_file": "train.jsonl",
        "nonmember_file": "val.jsonl",
        "text_template": "Question: {dialog[0][content]}\nAnswer: {dialog[1][content]}",
    })
    row = {"dialog": [{"content": "What treats hypertension?", "role": "user"}, {"content": "ACE inhibitors", "role": "assistant"}]}
    assert schema.row_to_text(row) == "Question: What treats hypertension?\nAnswer: ACE inhibitors"


def test_schema_with_text_field():
    schema = build_schema({
        "format": "jsonl",
        "member_file": "m.jsonl",
        "nonmember_file": "n.jsonl",
        "text_field": "note",
    })
    assert schema.row_to_text({"note": "clinical note text"}) == "clinical note text"


def test_schema_text_template_takes_precedence_over_text_field():
    schema = build_schema({
        "format": "csv",
        "member_file": "m.csv",
        "nonmember_file": "n.csv",
        "text_field": "note",
        "text_template": "{note} (templated)",
    })
    assert schema.row_to_text({"note": "hello"}) == "hello (templated)"


def test_schema_requires_field_or_template():
    with pytest.raises(ValueError, match="text_field.*text_template"):
        build_schema({"format": "csv", "member_file": "m.csv", "nonmember_file": "n.csv"})


def test_schema_defaults_format_to_csv():
    schema = build_schema({"member_file": "m.csv", "nonmember_file": "n.csv", "text_field": "note"})
    assert schema.file_format == "csv"


def test_schema_missing_row_key_in_template_yields_empty_string():
    schema = build_schema({
        "format": "csv", "member_file": "m.csv", "nonmember_file": "n.csv",
        "text_template": "{missing_field}",
    })
    assert schema.row_to_text({"other": "value"}) == ""
