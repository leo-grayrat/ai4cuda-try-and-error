"""Known usage in interrupted attempts must remain in cost accounting."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_repeats as summary


def summarize_rows(specs):
    cache = Path(__file__).resolve().parents[3]/'.cache'
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=cache) as folder:
        root = Path(folder)
        assert root.resolve().is_relative_to(cache.resolve())
        trials, records = [], {}
        for number, (state, complete, tokens, events) in enumerate(specs):
            identifier = f't{number:03d}'
            trial = {'id': identifier, 'condition': 'phase4', 'task': 'branch_features',
                     'permissions': False, 'checker': False}
            trials.append(trial)
            run = root/'runs'/identifier
            run.mkdir(parents=True)
            (run/'result.json').write_text(json.dumps({'state': state, 'final_evaluation': {}}))
            (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            records[identifier] = {'terminal_result_available': True, 'final_correctness': None,
                'protected_files_changed': [], 'usage_records_complete': complete,
                'usage': {'input_tokens': 300, 'cached_input_tokens': 200,
                          'uncached_input_tokens': 100, 'output_tokens': tokens},
                'seconds': 2, 'report_available': False, 'prose_available': False}
        with patch.object(summary, 'ROOT', root), patch.object(summary, 'collect', side_effect=lambda run: records[run.name]):
            return summary.summarize({'run_root': 'runs', 'trials': trials})


class PartialUsageTests(unittest.TestCase):
    def test_interrupted_known_tokens_contribute_without_claiming_complete_cost(self):
        result = summarize_rows([('api_or_trace_error', False, 50, [])])
        expected = (200*.003 + 100*.15 + 50*.6)/1e6
        self.assertAlmostEqual(result['conditions'][0]['all_attempt_known_cost_usd_sum'], expected)
        self.assertIsNone(result['trials'][0]['current_offpeak_estimate_usd'])
        self.assertTrue(result['conditions'][0]['cost_usage_incomplete'])

    def test_model_generated_median_excludes_external_interruption(self):
        result = summarize_rows([('submitted', True, 500, []), ('infrastructure_interrupted', False, 10, [])])
        self.assertEqual(result['conditions'][0]['generated_tokens_median'], 500)

    def test_natural_plain_final_rejection_is_counted(self):
        events = [{'type': 'plain_final_submission', 'index': 0,
                   'result': {'ok': False, 'error': 'Submission requires correctness and load measurements for the current code'}}]
        result = summarize_rows([('submitted', True, 50, events)])
        self.assertEqual(result['trials'][0]['submission_rejections'], 1)


if __name__ == '__main__':
    unittest.main()
