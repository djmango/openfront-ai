import copy
import unittest

from scripts.paired_policy_benchmark import EXPECTED_SCHEMAS, summarize, validate


def report(label, scores=(0.2, 0.8), maps=("Onion", "Pangaea")):
    return {
        "mode": "indirect-scripted-bot",
        "engine": "node-ts",
        "stage": 1,
        "max_ticks": 15000,
        "schema": EXPECTED_SCHEMAS[label],
        "episodes": [
            {
                "index": i,
                "seed": f"w{i}-ep0",
                "map": maps[i],
                "bots": 0,
                "difficulty": "Easy",
                "nations": 3,
                "decision_ticks": 15,
                "won": score == 1.0,
                "place": 1 if score == 1.0 else 2,
                "score": score,
            }
            for i, score in enumerate(scores)
        ],
    }


class PairedPolicyBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.reports = {
            "current_rust": report("current_rust", (1.0, 0.8)),
            "ppo_v5": report("ppo_v5"),
            "ppo_v7": report("ppo_v7", (0.5, 0.5)),
        }

    def test_summary_reports_paired_differences_and_intervals(self):
        result = summarize(self.reports)
        self.assertEqual(result["mode"], "indirect-scripted-bot")
        self.assertEqual(result["scenario_count"], 2)
        delta = result["paired_differences"]["current_rust-minus-ppo_v5"]
        self.assertAlmostEqual(delta["mean_score_delta"], 0.4)
        self.assertEqual(len(delta["mean_score_delta_ci95_bootstrap"]), 2)

    def test_mismatched_map_is_rejected(self):
        bad = copy.deepcopy(self.reports)
        bad["ppo_v7"]["episodes"][1]["map"] = "World"
        with self.assertRaisesRegex(ValueError, "scenario list differs"):
            validate(bad)

    def test_incompatible_schema_is_rejected(self):
        bad = copy.deepcopy(self.reports)
        bad["ppo_v5"]["schema"]["actions"] = 21
        with self.assertRaisesRegex(ValueError, "incompatible schema"):
            validate(bad)

    def test_engine_mismatch_is_rejected(self):
        bad = copy.deepcopy(self.reports)
        bad["ppo_v5"]["engine"] = "native-rust"
        with self.assertRaisesRegex(ValueError, "engine"):
            validate(bad)


if __name__ == "__main__":
    unittest.main()
