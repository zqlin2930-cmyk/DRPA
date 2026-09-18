"""CPU/AST checks: no torch import, no CUDA, no experiment execution."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent


class ExecutionContract(unittest.TestCase):
    def setUp(self):
        self.code = (ROOT / 'run_fullft_fewshot.py').read_text()
        self.tree = ast.parse(self.code)
        self.functions = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}

    def test_parameter_and_gradient_counts_are_separate(self):
        assignments = {n.targets[0].id: n.value.value for n in self.tree.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and isinstance(n.value, ast.Constant)}
        self.assertEqual(assignments['COUNT'], 440029541)
        self.assertEqual(assignments['LOSS_PATH'], 339291552)

    def test_no_inference_during_cpu_prepare(self):
        code = ast.get_source_segment(self.code, self.functions['prepare'])
        self.assertIn("model_setup(torch.device('cpu'))", code)
        self.assertNotIn('.backward(', code)
        self.assertNotIn('.step(', code)
        self.assertNotIn('evaluate_one(', code)

    def test_only_support_consumed_in_training(self):
        code = ast.get_source_segment(self.code, self.functions['train'])
        self.assertIn("PARENT/'support_manifest.csv'", code)
        self.assertIn("PARENT/'sample_order_500.csv'", code)
        self.assertNotIn('query_manifest', code)
        self.assertEqual(code.count('opt.step()'), 1)
        self.assertIn('(loss/8).backward()', code)
        self.assertNotIn('autocast(', code)
        self.assertNotIn('GradScaler(', code)
        self.assertIn('save_state(wrapper.model,opt,500,final=True)', code)

    def test_query_follows_frozen_original_evaluator(self):
        code = ast.get_source_segment(self.code, self.functions['evaluate'])
        self.assertIn("TARGET/'TRAIN_COMPLETE.json'", code)
        self.assertIn("PARENT/'query_manifest.csv'", code)
        self.assertIn("old.evaluate_one('FullFT'", code)
        self.assertIn('strict=True', code)
        self.assertIn('frame.lcc.eq(0).all()', code)

    def test_manager_has_exactly_one_train_then_query_then_report(self):
        tree = ast.parse((ROOT / 'launch_fullft_fewshot.py').read_text())
        stages = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                  and isinstance(n.target, ast.Name) and n.target.id == 'stage']
        self.assertEqual(len(stages), 1)
        self.assertEqual(ast.literal_eval(stages[0].iter), ['train', 'evaluate', 'summary'])


if __name__ == '__main__':
    unittest.main()
