"""Check event distinctions that change the interpretation of enforcement."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_enforcement import analyze_events, GATE_ERROR, PERMISSION_ERROR, STAGE_ERROR

BACKEND = 'flashrt/apps/branch_features/backend/'


def tool(index, operation, result, path=None):
    args = {'operation': operation}
    if path:
        args['path'] = path
    return {'type': 'tool', 'index': index, 'call': {'function': {'arguments': json.dumps(args)}}, 'result': result}


def response(index, tokens):
    return {'type': 'response', 'index': index, 'usage': {'completion_tokens': tokens}}


class EventInterpretationTests(unittest.TestCase):
    def test_permission_and_report_stage_are_different(self):
        rows = [tool(0, 'write', {'ok': False, 'error': PERMISSION_ERROR}, 'README.md'),
                tool(1, 'execute', {'ok': False, 'error': STAGE_ERROR}, BACKEND+'probe.py'),
                tool(2, 'write', {'ok': True}, './README.md'),
                tool(3, 'write', {'ok': True}, BACKEND+'../backend/worker.py')]
        result = analyze_events(rows, BACKEND)
        self.assertEqual(len(result['permission_denials']), 1)
        self.assertEqual(len(result['closed_development_stage_denials']), 1)
        self.assertEqual([r['path'] for r in result['successful_tool_edits_outside_assigned_backend']], ['./README.md'])
        self.assertEqual(result['gate_rejections'], [])

    def test_natural_plain_submission_and_post_feedback_actions(self):
        rows = [response(0, 100), {'type': 'plain_final_submission', 'index': 0,
                                  'result': {'ok': False, 'error': GATE_ERROR}},
                response(1, 37), tool(1, 'verify', {'ok': True}, BACKEND+'worker.py'),
                tool(1, 'benchmark', {'ok': True}, BACKEND+'worker.py'),
                response(2, 13), {'type': 'plain_final_submission', 'index': 2,
                                  'result': {'ok': True, 'submitted': True}}]
        result = analyze_events(rows, BACKEND)
        self.assertEqual(len(result['submission_events']), 2)
        self.assertEqual(len(result['gate_rejections']), 1)
        self.assertEqual(result['actor_verify_passes_before_first_submission'], 0)
        self.assertEqual(result['observed_api_responses_after_first_gate_rejection'], 2)
        self.assertEqual(result['observed_generated_tokens_after_first_gate_rejection'], 50)
        self.assertEqual(result['successful_tool_edits_after_first_gate_rejection'], [])

    def test_same_response_action_precedes_model_receiving_feedback(self):
        rows = [response(0, 100), tool(0, 'finish', {'ok': False, 'error': GATE_ERROR}),
                tool(0, 'write', {'ok': True}, BACKEND+'worker.py'),
                response(1, 37), tool(1, 'finish', {'ok': True, 'submitted': True})]
        result = analyze_events(rows, BACKEND)
        self.assertEqual(len(result['same_response_actions_after_first_gate_rejection']), 1)
        self.assertEqual(result['observed_generated_tokens_after_first_gate_rejection'], 37)
        self.assertEqual(result['successful_tool_edits_after_first_gate_rejection'][0]['response_index'], 0)

    def test_prior_evidence_does_not_imply_current_version_passes_gate(self):
        rows = [tool(0, 'verify', {'ok': True}, BACKEND+'old.py'),
                tool(1, 'benchmark', {'ok': True}, BACKEND+'old.py'),
                tool(2, 'finish', {'ok': False, 'error': GATE_ERROR})]
        result = analyze_events(rows, BACKEND)
        self.assertEqual(result['actor_verify_passes_before_first_submission'], 1)
        self.assertEqual(result['actor_measurements_completed_before_first_submission'], 1)
        self.assertEqual(len(result['gate_rejections']), 1)

    def test_truncated_unexecuted_call_is_not_a_submission(self):
        unexecuted = tool(0, 'finish', {'ok': False})
        unexecuted['type'] = 'truncated_call_not_executed'
        malformed = tool(1, 'finish', {'ok': False})
        malformed['call']['function']['arguments'] = '{'
        rows = [unexecuted, malformed, {'type': 'plain_final_submission', 'index': 2,
                                       'result': {'ok': True, 'submitted': True}}]
        result = analyze_events(rows, BACKEND)
        self.assertEqual(len(result['submission_events']), 1)
        self.assertEqual(result['actor_verify_passes_before_first_submission'], 0)
        self.assertEqual(result['gate_rejections'], [])


if __name__ == '__main__':
    unittest.main()
