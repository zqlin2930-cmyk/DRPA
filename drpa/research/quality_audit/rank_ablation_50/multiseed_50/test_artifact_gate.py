"""Read-only regression checks against the completed seed3407 artifacts."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import unittest
from unittest.mock import patch

import pandas as pd
import rank4_multiseed_50_supervisor as supervisor


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = supervisor.ROOT / "seed3407_r4"
        cls.rows = pd.read_csv(cls.run_dir / "validation_rows_step_06000.csv")

    def gate(self, rows):
        with patch.object(supervisor.pd, "read_csv", return_value=rows):
            return supervisor.artifact_gate(self.run_dir, 3407)

    def test_completed_run_passes(self):
        self.assertEqual(self.gate(self.rows.copy()), (True, "ARTIFACT_GATE_PASS"))

    def test_missing_prompt_blocks(self):
        ok, detail = self.gate(self.rows.drop(columns="prompt"))
        self.assertFalse(ok)
        self.assertIn("validation_schema_missing", detail)

    def test_duplicate_prompt_blocks(self):
        rows = self.rows.copy()
        group = rows.loc[rows.lcc.eq(0)].groupby("case_id").head(8)
        indices = group.loc[group.case_id.eq(group.iloc[0].case_id)].index
        rows.loc[indices[1], "prompt"] = rows.loc[indices[0], "prompt"]
        self.assertEqual(self.gate(rows), (False, "duplicate_validation_rows"))

    def test_unknown_prompt_blocks(self):
        rows = self.rows.copy()
        rows.loc[rows.index[0], "prompt"] = "unknown ROI"
        self.assertEqual(self.gate(rows), (False, "validation_prompt_coverage_mismatch"))

    def test_nonfinite_metric_blocks(self):
        rows = self.rows.copy()
        rows.loc[rows.index[0], "dice"] = float("nan")
        self.assertEqual(self.gate(rows), (False, "nonfinite_or_missing_validation_metric"))

    def test_completed_seed_is_not_retrained(self):
        with patch.object(supervisor.subprocess, "run", side_effect=AssertionError("retraining prohibited")) as launch:
            supervisor.run_one(3407)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
