"""
Unit tests for `logic_slm_proofwriter.py` (proof parsing and solutions).

Import as:

import test.test_logic_slm_proofwriter as ttlspw
"""

import os

import pytest

import logic_slm_format as lsform
import logic_slm_proofwriter as lspw

_THEORY = ("Anne is kind. Bob is furry. Green people are nice. All furry people are green. "
           "Kind people are furry. If someone is green and furry then they are not big.")
_PROOFS = ("@0: Anne is kind.[(triple1)] Bob is furry.[(triple2)] "
           "@1: Anne is furry.[(((triple1) -> rule3))] Bob is green.[(((triple2) -> rule2))] "
           "@2: Anne is green.[(((((triple1) -> rule3)) -> rule2))] "
           "Bob is nice.[(((((triple2) -> rule2)) -> rule1))] "
           "Bob is not big.[(((((triple2) -> rule2) triple2) -> rule4))] "
           "@3: Anne is nice.[(((((((triple1) -> rule3)) -> rule2)) -> rule1))] "
           "Anne is not big.[(((((((triple1) -> rule3)) -> rule2) ((triple1) -> rule3)) -> rule4))]")


def _row(question, answer):
    return {"theory": _THEORY, "question": question, "answer": answer, "allProofs": _PROOFS,
            "NRule": 4, "QDep": 2, "config": "depth-3"}


class Test_parse:
    def test_alternatives(self) -> None:
        alts = lspw.parse_proof("(triple2 OR ((triple3) -> rule3))")
        assert alts == [("T", 2), ("R", (("T", 3),), 3)]

    def test_multi_premise(self) -> None:
        (node,) = lspw.parse_proof("(((((triple2) -> rule2) triple2) -> rule4))")
        assert node == ("R", (("R", (("T", 2),), 2), ("T", 2)), 4)

    def test_negation(self) -> None:
        assert lspw.is_negation("The dog does not need the bear.", "The dog needs the bear.")
        assert lspw.is_negation("Anne is green.", "Anne is not green.")
        assert not lspw.is_negation("Anne is green.", "Anne is green.")
        assert not lspw.is_negation("Anne is not green.", "Bob is green.")


class Test_solutions:
    def test_yes(self) -> None:
        assert lspw.solution_steps(_row("Anne is green.", "True")) == [
            "Anne is kind, Kind people are furry => Anne is furry",
            "Anne is furry, All furry people are green => Anne is green",
        ]

    def test_no(self) -> None:
        steps = lspw.solution_steps(_row("Bob is big.", "False"))
        assert steps[-1] == ("Bob is green, Bob is furry, If someone is green and furry "
                             "then they are not big => Bob is not big")
        assert steps[0] == "Bob is furry, All furry people are green => Bob is green"

    def test_given_fact(self) -> None:
        assert lspw.solution_steps(_row("Bob is furry.", "True")) == ["Bob is furry"]

    def test_unknown_lists_closure_once(self) -> None:
        steps = lspw.solution_steps(_row("Anne is round.", "Unknown"))
        assert len(steps) == 7 and len(set(steps)) == 7

    def test_unknown_contradicted_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            lspw.solution_steps(_row("Anne is green.", "Unknown"))

    def test_rule_count_checked(self) -> None:
        row = {**_row("Anne is green.", "True"), "NRule": 5}
        with pytest.raises(ValueError):
            lspw.solution_steps(row)

    def test_example_renders(self) -> None:
        ex = lspw.to_example(_row("Anne is green.", "True"))
        text = lsform.render_prompt(ex) + lsform.render_solution(ex)
        assert text.startswith("Premises: Anne is kind.")
        assert "Question: Anne is green?\nSolution:\n" in text
        assert text.endswith("#### yes")


_RAW = "data/raw"


@pytest.mark.skipif(not os.path.exists(os.path.join(_RAW, "proofwriter/data")),
                    reason="ProofWriter not downloaded")
class Test_real_data:
    def test_sft_pairs_and_eval(self) -> None:
        recs = lspw.build_eval(_RAW, per_depth=20, seed=0)
        assert len(recs) == 120
        assert sorted({r["qdep"] for r in recs}) == [0, 1, 2, 3, 4, 5]
        excl = lspw.eval_theories(recs)
        pairs, val, counts = lspw.load_sft_pairs(_RAW, 300, 20, seed=0, exclude_theories=excl)
        assert len(pairs) == 300 and len(val) == 20
        assert counts["proofwriter_skipped"] <= 0.01 * 320
        for p, s in pairs:
            assert p.split("\n", 1)[0][len("Premises: "):] not in excl
            assert s.rsplit("\n", 1)[-1] in ("#### yes", "#### no", "#### unknown")
