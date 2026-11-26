import try_models


class Test_build_prompt:
    def test_math_problem(self) -> None:
        act = try_models.build_prompt("  Tom has 3  apples.\tHow many? ")
        assert act == "Problem: Tom has 3 apples. How many?\nSolution:\n"

    def test_problem_prefix_is_not_doubled(self) -> None:
        act = try_models.build_prompt("Problem: 2 + 3 = ?")
        assert act == "Problem: 2 + 3 = ?\nSolution:\n"

    def test_logic_problem(self) -> None:
        act = try_models.build_prompt("Premises: p -> q. p. | Question: q?")
        assert act == "Premises: p -> q. p.\nQuestion: q?\nSolution:\n"

    def test_logic_question_without_label(self) -> None:
        act = try_models.build_prompt("Premises: It rains. | Is it true that it rains")
        assert act == "Premises: It rains.\nQuestion: Is it true that it rains?\nSolution:\n"
