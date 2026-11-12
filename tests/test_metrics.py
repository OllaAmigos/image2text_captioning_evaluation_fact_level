import math
from captioneval.metrics import weighted_score, is_good, f1, critical_recall
from captioneval.facts import Fact, validate_caption_facts
from captioneval.rewrites import mixed_script_tokens, text_preservation, ellipsis_truncated


def test_weighted_score():
    items = [("full", "critical"), ("wrong", "minor"), ("partial", "important")]
    assert math.isclose(weighted_score(items), (5 + 0 + 1.5) / 9)
    assert weighted_score([]) is None


def test_good_rule():
    assert is_good(0.9, 0.6, False)
    assert not is_good(0.9, 0.6, True)
    assert not is_good(0.8, 0.9, False)


def test_f1_and_critical_recall():
    assert math.isclose(f1(0.5, 0.5), 0.5)
    assert critical_recall(["full", "partial", "missing"]) == 0.5


def test_gates_pass_and_fail():
    cap = "A red lighthouse stands on a rocky cliff"
    good = [Fact("f1", "A red lighthouse", "critical", span="A red lighthouse"),
            Fact("f2", "stands on a rocky cliff", "important", span="stands on a rocky cliff")]
    assert validate_caption_facts(cap, good).passed
    assert not validate_caption_facts(cap, None).passed
    bad = [Fact("f1", "A lighthouse", "important", span="made up span")]
    rep = validate_caption_facts(cap, bad)
    assert "no_critical_fact" in rep.failures


def test_rewrite_helpers():
    assert mixed_script_tokens("a terraceковой view") == ["terraceковой"]
    assert mixed_script_tokens("a clean caption") == []
    assert text_preservation(["OPEN"], "a sign saying \"open\"") == 1.0
    assert text_preservation(["OPEN"], "a shop sign") == 0.0
    assert ellipsis_truncated("the sign reads 'GRAND OPE...'")
