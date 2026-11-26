import gzip
import json

import baseline_eval as be


class Test_trim_completion:
    def test_cuts_after_the_answer_line(self) -> None:
        text = "3 + 4 = 7\n#### 7\n\nProblem: next one\nSolution:\n1 + 1 = 2\n#### 2"
        assert be.trim_completion(text) == "3 + 4 = 7\n#### 7"

    def test_cuts_before_a_new_problem_without_an_answer(self) -> None:
        assert be.trim_completion("3 + 4 = 7\n\nProblem: next\n#### 9") == "3 + 4 = 7"
        assert be.trim_completion("a => b\nPremises: x.\n#### yes") == "a => b"

    def test_keeps_text_without_answer(self) -> None:
        assert be.trim_completion(" 3 + 4 = 7 \n") == "3 + 4 = 7"


class Test_fewshot_prefix:
    def test_examples_are_separated_by_blank_lines(self) -> None:
        shots = [("Problem: a\nSolution:\n", "1\n#### 1"), ("Problem: b\nSolution:\n", "2\n#### 2\n")]
        act = be.fewshot_prefix(shots)
        assert act == "Problem: a\nSolution:\n1\n#### 1\n\nProblem: b\nSolution:\n2\n#### 2\n\n"
        # The trimmed continuation of a prefixed prompt is scored like a normal completion.
        assert be.trim_completion("3\n#### 3\n\nProblem: c") == "3\n#### 3"


class Test_balanced_shots:
    def test_typical_length_per_answer_interleaved(self) -> None:
        pairs = [("P long long\n", "s\n#### yes"), ("P\n", "s\n#### yes"), ("Pn\n", "s\n#### no"),
                 ("Pu\n", "#### unknown"), ("P medium\n", "s\n#### yes")]
        act = be.balanced_shots(pairs, 2)
        answers = [s.split("#### ")[1] for _, s in act]
        assert answers == ["no", "unknown", "yes", "yes"]
        # Of the three "yes" examples the two from the median upwards are used, not the shortest.
        assert act[2][0] == "P medium\n" and act[3][0] == "P long long\n"

    def test_one_per_answer_is_the_median(self) -> None:
        pairs = [("a\n", "#### yes"), ("aaa\n", "#### yes"), ("aaaaa\n", "#### yes")]
        assert be.balanced_shots(pairs, 1) == [("aaa\n", "#### yes")]


class Test_math_shots:
    def test_reads_first_n(self, tmp_path) -> None:
        with gzip.open(tmp_path / "part-00000.jsonl.gz", "wt") as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"Problem: {i}\nSolution:\n", "solution": f"#### {i}"}) + "\n")
        act = be.math_shots(str(tmp_path), 3)
        assert [s for _, s in act] == ["#### 0", "#### 1", "#### 2"]
