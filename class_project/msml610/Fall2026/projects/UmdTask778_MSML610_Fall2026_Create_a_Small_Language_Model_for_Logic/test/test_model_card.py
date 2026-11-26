import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))

import model_card  # noqa: E402


def _cost_file(tmp_path) -> str:
    path = tmp_path / "training_cost.json"
    stage = lambda su: {"su": su, "gpu_hours": su / 144, "jobs": 1}  # noqa: E731
    with open(path, "w") as f:
        json.dump({"models": {"1M": {"pretrain": stage(57.6), "sft_fixes": stage(129.6), "rft": stage(72.0),
                                     "grpo": stage(72.0)}}}, f)
    return str(path)


class Test_lineage_cost_table:
    def test_grpo_lists_all_four_stages_and_the_total(self, tmp_path) -> None:
        act = model_card.lineage_cost_table("q1m", "grpo", _cost_file(tmp_path))
        rows = act.split("\n")
        assert [r.split("|")[1].strip() for r in rows[2:]] == ["Pretraining", "SFT + fixes", "RFT", "GRPO", "**Total**"]
        assert "| **Total** | **2.3** | **331** |" in act

    def test_earlier_stage_stops_early(self, tmp_path) -> None:
        act = model_card.lineage_cost_table("q1m", "sft-fixes", _cost_file(tmp_path))
        assert "RFT" not in act and "SFT + fixes" in act

    def test_unknown_size_or_stage_or_file(self, tmp_path) -> None:
        path = _cost_file(tmp_path)
        assert model_card.lineage_cost_table("q10m", "grpo", path) == ""
        assert model_card.lineage_cost_table("q1m", "sft", path) == ""
        assert model_card.lineage_cost_table("q1m", "grpo", str(tmp_path / "missing.json")) == ""


class Test_build_card:
    def test_every_size_builds_a_card(self) -> None:
        for size in model_card.SIZES:
            card = model_card.build_card(size=size, stage="grpo", eval_run="", license_id="cc-by-4.0",
                                         repo_id=f"user/slm-math-{size}")
            assert card.startswith("---\nlicense: cc-by-4.0\n")
            assert f'from_pretrained("user/slm-math-{size}")' in card
            assert "GRPO reinforcement learning" in card
            assert "see dataset card" not in card


class Test_family:
    def test_family_card_has_nickname_title_and_all_six_models(self) -> None:
        card = model_card.build_card(size="q125m", stage="grpo", eval_run="", license_id="cc-by-4.0",
                                     repo_id="user/logi-robin-119M", family=True)
        assert "\n# Logi-Robin (~119M parameters)\n" in card
        assert 'from_pretrained("user/logi-robin-119M")' in card
        assert "| Logi-Robin (this model) | ~119M | `user/logi-robin-119M` |" in card
        for repo, _ in model_card.FAMILY.values():
            assert f"`user/{repo}`" in card

    def test_family_covers_every_size(self) -> None:
        assert set(model_card.FAMILY) == set(model_card.SIZES)
