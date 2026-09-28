import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_archived_source_retains_hash_and_is_not_an_active_entry(self):
        manifest = json.loads((ROOT / 'archive/manifest.json').read_text())
        for row in manifest['files']:
            archived = ROOT / row['archived_path']
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), row['sha256'])
            self.assertFalse((ROOT / row['original_path']).exists(), row['original_path'])
            if row['original_path'].startswith('scripts/'):
                self.assertEqual(archived.suffix, '.txt')
        for name in ('run_checks.sh', 'run_repeats.sh'):
            self.assertFalse((ROOT / 'scripts' / name).exists())

    def test_ppo_template_is_available_without_local_experiment_artifacts(self):
        config = json.loads((ROOT / 'configs/ppo.json').read_text())
        self.assertEqual(config['algorithm']['gamma'], .995)
        self.assertEqual(config['algorithm']['lam'], .95)
        self.assertIn('actor', config)
        self.assertIn('critic', config)

    def test_documented_public_entries_exist(self):
        for name in ('run_local.sh', 'run_contact.py', 'run_probe.py',
                     'spike013_v02_v03.py', 'evaluate_checkpoint.py', 'validate_workspace.py'):
            self.assertTrue((ROOT / 'scripts' / name).is_file(), name)
        for name in ('PLAN.md', 'CONICAL_PROBE_PLAN.md', 'REPORT.md'):
            self.assertFalse((ROOT / name).exists())
            self.assertTrue((ROOT / 'docs/history' / name).is_file())


if __name__ == '__main__':
    unittest.main()
