import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import evaluate_model
import score_audit


class PragmaticEvaluationTests(unittest.TestCase):
    def test_incremental_and_wasted_spend_are_resolution_aware(self):
        rows = [
            {"id": 0, "label": "long", "prompt": "first this then that"},
            {"id": 1, "label": "medium", "prompt": "a normal action"},
        ]
        predictions = [
            {
                "duration": 6,
                "duration_source": "model",
                "confidence": 0.9,
                "estimated_cost_microusd": 300_000,
            },
            {
                "duration": 6,
                "duration_source": "model",
                "confidence": 0.9,
                "estimated_cost_microusd": 420_000,
            },
        ]
        result = evaluate_model.metrics(rows, predictions)
        self.assertEqual(result["per_class"]["short"]["accepted"], 0)
        self.assertEqual(result["per_class"]["long"]["accepted"], 2)
        self.assertEqual(result["per_class"]["long"]["correct"], 1)
        self.assertEqual(result["incremental_spend_microusd"], 240_000)
        self.assertEqual(result["wasted_incremental_spend_microusd"], 140_000)

    def test_pragmatic_audit_accepts_exactly_seventy_five_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "expected.jsonl"
            judgments = root / "judgments.jsonl"
            output = root / "report.json"
            expected_rows = []
            judgment_rows = []
            for row_id in range(200):
                accepted = row_id < 20
                expected_rows.append(
                    {
                        "id": row_id,
                        "audit_slice": (
                            "accepted_override" if accepted else "natural"
                        ),
                        "duration": 6 if accepted else 4,
                    }
                )
                judgment_rows.append(
                    {
                        "id": row_id,
                        "final_label": (
                            "long"
                            if row_id < 15
                            else "medium"
                        ),
                    }
                )
            expected.write_text(
                "".join(json.dumps(row) + "\n" for row in expected_rows),
                encoding="utf-8",
            )
            judgments.write_text(
                "".join(json.dumps(row) + "\n" for row in judgment_rows),
                encoding="utf-8",
            )
            argv = sys.argv
            try:
                sys.argv = [
                    "score_audit.py",
                    "--expected",
                    str(expected),
                    "--judgments",
                    str(judgments),
                    "--output",
                    str(output),
                    "--release-policy",
                    "long-only-pragmatic-v2",
                ]
                score_audit.main()
            finally:
                sys.argv = argv
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["accepted_override_examples"], 20)
            self.assertEqual(report["accepted_override_agreement"], 0.75)
            self.assertEqual(report["severe_short_under_duration_errors"], 0)


if __name__ == "__main__":
    unittest.main()
