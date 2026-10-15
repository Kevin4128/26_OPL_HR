"""Run with the ROS environment sourced: python3 -m unittest discover -s test."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from lifecycle_msgs.msg import Transition

spec = importlib.util.spec_from_file_location(
    'activator', Path(__file__).parents[1] / 'scripts/lifecycle_activator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ActivationOrderTest(unittest.TestCase):
    def test_all_models_configure_before_any_activation(self):
        calls = []

        class FakeActivator:
            nodes_to_activate = ['detector', 'tracker', 'recognizer']

            def transition_node(self, name, transition, label):
                calls.append((name, transition))
                return name != 'tracker'

        with patch.object(module.time, 'sleep'):
            module.LifecycleActivator.activate_all(FakeActivator())
        self.assertEqual(calls, [
            ('detector', Transition.TRANSITION_CONFIGURE),
            ('tracker', Transition.TRANSITION_CONFIGURE),
            ('recognizer', Transition.TRANSITION_CONFIGURE),
            ('detector', Transition.TRANSITION_ACTIVATE),
            ('recognizer', Transition.TRANSITION_ACTIVATE),
        ])
