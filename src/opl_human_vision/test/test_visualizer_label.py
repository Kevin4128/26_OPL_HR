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

    def test_predicted_label_and_exact_display_pair(self):
        human = SimpleNamespace(raw_track_id=1, track_id=7, name='Alice', confidence=.9, observed=False)
        self.assertIn('[predicted]', module.person_label(human))
        from collections import OrderedDict
        view = module.PipelineVisualizer.__new__(module.PipelineVisualizer)
        def message(stamp, frame='rgb'):
            return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp,nanosec=0),frame_id=frame),humans=[human])
        rgb = message(10); latest = message(11)
        view.images = OrderedDict([(view.frame_key(rgb),rgb)])
        view.latest_image_msg = latest
        view.latest_human_msg = message(10,'/rgb')
        view.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10500000000))
        self.assertEqual(view.matched_display(), (rgb,[human]))
        view.latest_human_msg = message(10,'wrong')
        self.assertEqual(view.matched_display(), (latest,[]))
        view.latest_human_msg = message(9)
        self.assertEqual(view.matched_display(), (latest,[]))
