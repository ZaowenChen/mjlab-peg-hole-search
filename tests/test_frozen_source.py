import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from mjlab_contact_prep.frozen_source import (
    frozen_hashes, materialize_frozen_source, resolve_frozen_source,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class FrozenSourceTests(unittest.TestCase):
    def test_changed_live_source_uses_original_tar_bytes_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = 'src/mjlab_contact_prep/controller.py'
            live = root / name
            live.parent.mkdir(parents=True)
            live.write_bytes(b'new development implementation')
            (root / 'reference').mkdir()
            original = b'original frozen implementation'
            with tarfile.open(root / 'reference/source.tar.gz', 'w:gz') as archive:
                member = tarfile.TarInfo(name)
                member.size = len(original)
                archive.addfile(member, io.BytesIO(original))
            plan = {'physics_frozen': {name: digest(original)}}
            resolved = resolve_frozen_source(plan, root, root / 'experiment')
            self.assertEqual(resolved[name][0], original)
            self.assertEqual(live.read_bytes(), b'new development implementation')
            self.assertEqual(plan['physics_frozen'][name], digest(original))
            with self.assertRaises(ValueError):
                resolve_frozen_source({'physics_frozen': {name: digest(b'unknown')}}, root, root / 'experiment')

    def test_per_run_snapshot_and_archived_paths_keep_original_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / 'experiment'
            saved = experiment / 'source/configs/old.json'
            saved.parent.mkdir(parents=True)
            saved.write_bytes(b'{"original":true}')
            archived = root / 'archive/scripts/old.py.txt'
            archived.parent.mkdir(parents=True)
            archived.write_bytes(b'original script')
            (root / 'archive/manifest.json').write_text(json.dumps({'files': [
                {'original_path': 'scripts/old.py', 'archived_path': 'archive/scripts/old.py.txt'}]}))
            plan = {'experiment_source_sha256': {
                'configs/old.json': digest(saved.read_bytes()),
                'scripts/old.py': digest(archived.read_bytes()),
            }}
            resolved = resolve_frozen_source(plan, root, experiment)
            self.assertEqual(set(resolved), set(plan['experiment_source_sha256']))
            self.assertFalse((root / 'scripts/old.py').exists())

    def test_replay_tree_isolated_source_and_ppo_template_leave_data_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'
            (root / 'src').mkdir(parents=True)
            (root / 'src/live.py').write_bytes(b'current')
            config = root / 'evaluation/gpu_ppo_pilot_v2/amp_0/config.json'
            config.parent.mkdir(parents=True)
            config.write_bytes(b'current template')
            (root / 'reference').mkdir()
            files = {'src/live.py': (b'original', 'snapshot'),
                     'evaluation/gpu_ppo_pilot_v2/amp_0/config.json': (b'original template', 'snapshot')}
            destination = Path(tmp) / 'replay'
            materialize_frozen_source(files, root, destination)
            self.assertEqual((destination / 'src/live.py').read_bytes(), b'original')
            self.assertEqual((destination / config.relative_to(root)).read_bytes(), b'original template')
            self.assertEqual(config.read_bytes(), b'current template')
            self.assertEqual((root / 'src/live.py').read_bytes(), b'current')
            with self.assertRaises(FileExistsError):
                materialize_frozen_source(files, root, destination)

    def test_missing_empty_conflicting_and_unsafe_source_are_rejected(self):
        for plan in ({}, {'physics_frozen': {'../escape.py': digest(b'bad')}},
                     {'physics_frozen': {'/absolute.py': digest(b'bad')}},
                     {'physics_frozen': {'src/x.py': digest(b'a')},
                      'experiment_source_sha256': {'src/x.py': digest(b'b')}}):
            with self.assertRaises(ValueError):
                frozen_hashes(plan)


if __name__ == '__main__':
    unittest.main()
