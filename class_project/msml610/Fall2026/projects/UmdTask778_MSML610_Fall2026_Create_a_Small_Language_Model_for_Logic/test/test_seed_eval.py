import pytest

import seed_eval as se

RECS = [{"task": "math", "answer": "7"}, {"task": "math", "answer": "3"}]


class Test_majority_answer:
    def test_most_common_normalized_answer(self) -> None:
        comps = ["x\n#### 7", "y\n#### 7.0", "z\n#### 8", "no answer here"]
        assert se.majority_answer(comps, "math") == "7"

    def test_no_answer(self) -> None:
        assert se.majority_answer(["nothing", ""], "math") == ""


class Test_sample_metrics:
    def test_pass_and_majority(self) -> None:
        samples = [["#### 7", "#### 7", "#### 8", "#### 9"],   # 2 of 4 correct, majority correct
                   ["#### 4", "#### 4", "#### 3", "junk"]]      # 1 of 4 correct, majority wrong
        m = se.sample_metrics(RECS, samples)
        assert m["pass@1"] == pytest.approx((0.5 + 0.25) / 2)
        assert m["pass@4"] == 1.0
        assert m["majority@4"] == 0.5


class Test_aggregate:
    def test_mean_std_range(self) -> None:
        agg = se.aggregate({0: {"pass@1": 0.4}, 1: {"pass@1": 0.5}, 2: {"pass@1": 0.6}})
        assert agg["pass@1"]["mean"] == pytest.approx(0.5)
        assert agg["pass@1"]["std"] == pytest.approx(0.1)
        assert (agg["pass@1"]["min"], agg["pass@1"]["max"], agg["pass@1"]["n_seeds"]) == (0.4, 0.6, 3)

    def test_single_seed_has_zero_std(self) -> None:
        assert se.aggregate({0: {"pass@1": 0.4}})["pass@1"]["std"] == 0.0
