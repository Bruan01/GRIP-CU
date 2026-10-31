"""CPU checks that watchdog recovery never duplicates or expands GPU work."""
import fcntl
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_stage2_queue as watch


class WatchdogTests(unittest.TestCase):
    def test_healthy_controller_continues_without_restart(self):
        self.assertEqual(watch.decide({}, True, False, [], {}, 0, 3), 'monitoring')

    def test_existing_worker_lock_prevents_duplicate_launch(self):
        self.assertEqual(watch.decide({}, False, True, [], {}, 0, 3), 'waiting_for_existing_work')

    def test_existing_gpu_process_prevents_duplicate_launch(self):
        self.assertEqual(watch.decide({}, False, False, ['123'], {}, 0, 3), 'waiting_for_existing_work')

    def test_finished_budget_without_gain_does_not_expand_search(self):
        state = {'action': 'search_budget_finished_targets_not_all_achieved'}
        self.assertEqual(watch.decide(state, False, False, [], {}, 0, 3), 'budget_finished')

    def test_unresolved_failure_can_resume_but_retry_limit_is_bounded(self):
        state = {'action': 'search_budget_finished_targets_not_all_achieved'}
        failures = {'nell23k': {'phase': 'search', 'error': 'fixture'}}
        self.assertEqual(watch.decide(state, False, False, [], failures, 0, 3), 'resume_queue')
        self.assertEqual(watch.decide(state, False, False, [], failures, 3, 3), 'retry_limit_reached')

    def test_alive_controller_finishes_other_work_before_failed_dataset_retry(self):
        self.assertEqual(watch.decide({}, True, True, ['123'], {'nell23k': {}}, 0, 3), 'monitoring')

    def test_lock_probe_does_not_release_existing_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'lock'
            with path.open('a+') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(watch.lock_busy(path))
                self.assertTrue(watch.lock_busy(path))
            self.assertFalse(watch.lock_busy(path))

    def test_unrelated_pid_is_not_a_controller(self):
        self.assertFalse(watch.controller_alive(os.getpid()))


if __name__ == '__main__':
    unittest.main()
