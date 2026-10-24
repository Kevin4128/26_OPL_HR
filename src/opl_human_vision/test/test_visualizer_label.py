import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    'visualizer', Path(__file__).parents[1] / 'scripts/visualizer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VisualizerLabelTest(unittest.TestCase):
    def test_session_id_does_not_replace_enrollment_id(self):
        human = SimpleNamespace(raw_track_id=1, track_id=1000000000,
                                name='unknown', confidence=0.95)
        self.assertEqual(module.person_label(human), 'ID:1 | unknown (0.95)')
        human.track_id = 7
        human.name = 'Alice'
        self.assertEqual(module.person_label(human), 'ID:7 (raw:1) | Alice (0.95)')
        human.raw_track_id = 9
        self.assertEqual(module.person_label(human), 'ID:7 (raw:9) | Alice (0.95)')

    def test_legacy_message_fallback(self):
        human = SimpleNamespace(raw_track_id=0, track_id=2,
                                name='Bob', confidence=0.9)
        self.assertEqual(module.person_label(human), 'ID:2 (raw:2) | Bob (0.90)')

    def test_retained_and_uncertain_labels(self):
        human = SimpleNamespace(raw_track_id=4, track_id=2, name='furia',
                                confidence=0.9, identity_state=2)
        self.assertIn('furia [tracked]', module.person_label(human))
        human.identity_state = 3
        self.assertEqual(module.person_label(human), 'ID:4 | identity uncertain (0.90)')
