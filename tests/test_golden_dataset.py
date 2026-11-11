"""Guards the golden dataset itself. No model or API calls."""

import json
from pathlib import Path

from evals.common import load_golden

KB = json.loads(Path("data/troubleshooting_guide.json").read_text(encoding="utf-8"))
KB_ISSUES = {item["issue"] for item in KB}
GOLDEN = load_golden()


def test_size_is_30_to_40():
    assert 30 <= len(GOLDEN) <= 40


def test_ids_unique_and_fields_present():
    ids = [g["id"] for g in GOLDEN]
    assert len(ids) == len(set(ids))
    for g in GOLDEN:
        assert g["question"].strip() and g["reference"].strip()
        assert g["category"] in {"single", "multi", "ambiguous", "out_of_scope"}


def test_relevant_issues_exist_in_knowledge_base():
    for g in GOLDEN:
        assert set(g["relevant_issues"]) <= KB_ISSUES, g["id"]


def test_out_of_scope_has_no_relevant_issues():
    for g in GOLDEN:
        assert (g["category"] == "out_of_scope") == (not g["relevant_issues"]), g["id"]


def test_every_kb_issue_is_covered():
    covered = {issue for g in GOLDEN for issue in g["relevant_issues"]}
    assert covered == KB_ISSUES
