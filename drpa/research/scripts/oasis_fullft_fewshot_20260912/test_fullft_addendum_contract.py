
# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import copy
import csv
from pathlib import Path
import tempfile
import unittest
from fullft_addendum_contract import prepare, validate_rows, HASHES

SOURCE = Path(__file__).resolve().parents[2] / 'outputs/oasis_fewshot_20260912'


class ContractTests(unittest.TestCase):
    def rows(self):
        result = []
        for name in ['support_manifest.csv', 'query_manifest.csv', 'sample_order_500.csv']:
            with (SOURCE/name).open(newline='') as f:
                result.append(list(csv.DictReader(f)))
        return result

    def test_unchanged_parent(self):
        result = prepare(SOURCE)
        import json
        original = json.loads((SOURCE/'FROZEN_CONFIG.json').read_text())
        matched = result['protocol']
        self.assertEqual({k:v for k,v in original.items() if k != 'models'},
                         {k:v for k,v in matched.items() if k != 'models'})
        self.assertEqual(matched['models'], {'FullFT': 440029541})
        self.assertEqual(result['new_run_count'], 1)

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in HASHES:
                (Path(temp)/name).write_bytes((SOURCE/name).read_bytes())
            with (Path(temp)/'query_manifest.csv').open('ab') as f:
                f.write(b'\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                prepare(temp)

    def test_overlap_rejected(self):
        s,q,o = self.rows()
        q[0]['case_id'] = s[0]['case_id']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_rows(s,q,o)

    def test_missing_step_rejected(self):
        s,q,o = self.rows()
        with self.assertRaisesRegex(ValueError, '500 consecutive'):
            validate_rows(s,q,o[:-1])

    def test_query_in_training_order_rejected(self):
        s,q,o = self.rows()
        o[0]['case_id'] = q[0]['case_id']
        with self.assertRaisesRegex(ValueError, 'case/index'):
            validate_rows(s,q,o)


if __name__ == '__main__':
    unittest.main()
