from __future__ import annotations

import pytest

from privaudit.data.schema import build_schema

# ---------------------------------------------------------------------------
# load_texts(): a wrong `format` for the actual file content should fail with
# a clear, actionable message instead of a raw pandas parser traceback.
# ---------------------------------------------------------------------------

def test_load_texts_csv_format_on_a_jsonl_file_gives_an_actionable_error(tmp_path):
    # Real JSONL rows vary in embedded-comma count line to line (longer dialog turns, etc.),
    # which is exactly what makes pandas' CSV parser choke with a field-count mismatch -- a
    # uniform/short fake file wouldn't reproduce it.
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(
        '{"dialog": [{"role": "user", "content": "a"}]}\n'
        '{"dialog": [{"role": "user", "content": "a"}]}\n'
        '{"dialog": [{"role": "user", "content": "a"}]}\n'
        '{"dialog": [{"role": "user", "content": "a, b, c, d, e, f, g, h"}, {"role": "assistant", "content": "z"}]}\n',
        encoding="utf-8",
    )
    schema = build_schema({"format": "csv", "member_file": "train.jsonl", "nonmember_file": "x", "text_field": "note"})

    with pytest.raises(ValueError, match="looks like JSONL"):
        schema.load_texts(tmp_path, "train.jsonl")


def test_load_texts_jsonl_format_on_a_csv_file_gives_an_actionable_error(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("note,label\nhello,0\nworld,1\n", encoding="utf-8")
    schema = build_schema({"format": "jsonl", "member_file": "train.csv", "nonmember_file": "x", "text_field": "note"})

    with pytest.raises(ValueError, match="doesn't look like JSONL"):
        schema.load_texts(tmp_path, "train.csv")


def test_load_texts_csv_happy_path_still_works(tmp_path):
    (tmp_path / "train.csv").write_text("note\nhello\nworld\n", encoding="utf-8")
    schema = build_schema({"format": "csv", "member_file": "train.csv", "nonmember_file": "x", "text_field": "note"})
    assert schema.load_texts(tmp_path, "train.csv") == ["hello", "world"]


def test_load_texts_jsonl_happy_path_still_works(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"note": "hello"}\n{"note": "world"}\n', encoding="utf-8")
    schema = build_schema({"format": "jsonl", "member_file": "train.jsonl", "nonmember_file": "x", "text_field": "note"})
    assert schema.load_texts(tmp_path, "train.jsonl") == ["hello", "world"]


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
