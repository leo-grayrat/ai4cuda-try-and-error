"""Review evidence gates do not replace adjudication of judgment accuracy."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_packet import validate


def fixture(report=True):
    packet = {"review_id": "opaque", "report_available": report,
              "submitted_report": "The kernel passed the final test." if report else None,
              "reference_model": "ref", "task_contract": "contract", "final_timing": None,
              "state": "submitted" if report else "generation_truncated", "final_sources": {},
              "changed_infrastructure": [], "source_findings": {},
              "visible_timeline": [{"id": "v0000", "text": "The kernel passed the final test."}],
              "independent_final_evaluation": {}, "rubric_dimensions": [{"id": "connected_explanation"}]}
    rules = [{"id": "C23"}]
    review = {"suspected_unmasking": None, "review_id": "opaque", "report_available": report,
              "prose_scores": {"connected_explanation": {"score": 1 if report else None,
                  "rationale": "This is a synthetic validation fixture.",
                  "evidence": [{"source": "report", "quote": "passed the final test"}] if report else []}},
              "rule_observations": [{"rule_id": "C23", "assessment": "insufficient",
                  "rationale": "The fixture is deliberately incomplete.", "evidence": []}]}
    return packet, rules, review


def test_fabricated_quote_or_wrong_source_cannot_pass_structural_review():
    packet, rules, review = fixture()
    validate(review, packet, rules)
    bad = deepcopy(review)
    bad["prose_scores"]["connected_explanation"]["evidence"][0]["quote"] = "invented faster measurement"
    with pytest.raises(ValueError, match="Unverifiable"):
        validate(bad, packet, rules)
    bad["prose_scores"]["connected_explanation"]["evidence"][0] = {"source": "unknown", "quote": "passed"}
    with pytest.raises(ValueError, match="Unverifiable"):
        validate(bad, packet, rules)


def test_absent_report_is_unscored_even_if_interim_message_exists():
    packet, rules, review = fixture(report=False)
    validate(review, packet, rules)
    review["prose_scores"]["connected_explanation"]["score"] = 0
    with pytest.raises(ValueError, match="fabricated prose score"):
        validate(review, packet, rules)


def test_duplicate_rules_and_definite_claim_without_evidence_are_rejected():
    packet, rules, review = fixture()
    review["rule_observations"].append(deepcopy(review["rule_observations"][0]))
    with pytest.raises(ValueError, match="duplicated"):
        validate(review, packet, rules)
    review["rule_observations"].pop()
    review["rule_observations"][0]["assessment"] = "observed"
    with pytest.raises(ValueError, match="no quoted evidence"):
        validate(review, packet, rules)


def test_actual_json_field_quotations_are_accepted_without_allowing_changed_values():
    packet, rules, review = fixture()
    packet["final_timing"] = {"median_us": {"candidate": 1.0}}
    review["rule_observations"][0].update(assessment="observed", evidence=[{
        "source": "final_timing", "quote": '"candidate": 1.0'}])
    validate(review, packet, rules)
    review["rule_observations"][0]["evidence"][0]["quote"] = '"candidate": 0.1'
    with pytest.raises(ValueError, match="Unverifiable"):
        validate(review, packet, rules)
