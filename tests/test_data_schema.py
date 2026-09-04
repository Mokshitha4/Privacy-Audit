from __future__ import annotations

import pytest

from privaudit.data.schema import KNOWN_SCHEMAS, build_custom_schema, resolve_schema


def test_medqa_row_to_text():
    schema = KNOWN_SCHEMAS["MedQA"]
    text = schema.row_to_text({"question": "What treats hypertension?", "answer": "ACE inhibitors"})
    assert text == "Question: What treats hypertension?\nAnswer: ACE inhibitors"


def test_icd_row_to_text_includes_note_and_codes():
    schema = KNOWN_SCHEMAS["ICD"]
    text = schema.row_to_text({
        "text_modified": "Patient presents with chest pain.",
        "diagnoses": "[{'diagnosis': 'chest pain', 'code': 'R07.9'}]",
    })
    assert "Patient presents with chest pain." in text
    assert "chest pain corresponds to R07.9" in text


def test_icd_row_to_text_handles_missing_codes():
    schema = KNOWN_SCHEMAS["ICD"]
    text = schema.row_to_text({"text_modified": "Routine visit.", "diagnoses": None})
    assert "No codes available" in text


def test_mimic_task_row_to_text_joins_input_and_output():
    schema = KNOWN_SCHEMAS["mortality"]
    text = schema.row_to_text({"input": "60yo male admitted with sepsis.", "output": "Mortality risk: high."})
    assert text == "60yo male admitted with sepsis.\n\nMortality risk: high."


def test_resolve_schema_known_dataset():
    assert resolve_schema("MedQA", None) is KNOWN_SCHEMAS["MedQA"]


def test_resolve_schema_custom_requires_schema_block():
    with pytest.raises(ValueError, match="no data.schema"):
        resolve_schema("custom", None)


def test_resolve_schema_unknown_dataset_name():
    with pytest.raises(ValueError, match="Unknown known_dataset"):
        resolve_schema("not_a_real_dataset", None)


def test_custom_schema_with_text_template():
    schema = build_custom_schema({
        "format": "csv",
        "member_file": "train.csv",
        "nonmember_file": "test.csv",
        "text_template": "Q: {question}\nA: {answer}",
    })
    assert schema.row_to_text({"question": "Dose?", "answer": "10mg"}) == "Q: Dose?\nA: 10mg"


def test_custom_schema_with_text_field():
    schema = build_custom_schema({
        "format": "jsonl",
        "member_file": "m.jsonl",
        "nonmember_file": "n.jsonl",
        "text_field": "note",
    })
    assert schema.row_to_text({"note": "clinical note text"}) == "clinical note text"


def test_custom_schema_requires_field_or_template():
    with pytest.raises(ValueError, match="text_field.*text_template"):
        build_custom_schema({"format": "csv", "member_file": "m.csv", "nonmember_file": "n.csv"})
