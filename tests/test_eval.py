"""Test cac ham cham diem thuan tuy trong eval (khong can model/mang)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("eg", ROOT / "eval" / "eval_generation.py")
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)


def test_f1_identical_is_one():
    assert eg.token_f1("neutrino oscillation dune", "dune neutrino oscillation") == 1.0


def test_f1_disjoint_is_zero():
    assert eg.token_f1("cat dog bird", "quantum accelerator physics") == 0.0


def test_f1_partial_between():
    f = eg.token_f1("dune neutrino beam", "dune neutrino oscillation")
    assert 0.0 < f < 1.0


def test_build_messages_rag_includes_context():
    chunks = [{"title": "DUNE", "text": "neutrino beam"}]
    msgs = eg.build_messages("What is DUNE?", chunks)
    assert msgs[0]["role"] == "system"
    assert "Context:" in msgs[-1]["content"]
    assert "neutrino beam" in msgs[-1]["content"]


def test_build_messages_norag_is_plain():
    msgs = eg.build_messages("What is DUNE?", [])
    assert "Context:" not in msgs[-1]["content"]
    assert msgs[-1]["content"] == "What is DUNE?"
