import json
import tempfile
import unittest
from pathlib import Path

from mjlab_contact_prep.configuration import resolve_config, write_effective_config


class ConfigurationTests(unittest.TestCase):
    def test_resolution_order_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            default = root / "default.json"
            experiment = root / "experiment.json"
            default.write_text(json.dumps({"format_version": 1, "a": {"x": 1, "y": 2}}))
            experiment.write_text(json.dumps({"format_version": 1, "a": {"y": 3}}))
            resolved = resolve_config(default, experiment, ["a.x=4", "flag=true"])
            self.assertEqual(resolved["effective"]["a"], {"x": 4, "y": 3})
            self.assertIs(resolved["effective"]["flag"], True)
            self.assertEqual(resolved["sources"]["overrides"], ["a.x=4", "flag=true"])

    def test_generator_overrides_are_retained_in_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            default = Path(folder) / "default.json"
            default.write_text('{"format_version": 1}')
            resolved = resolve_config(default, overrides=(item for item in ["x=1"]))
            self.assertEqual(resolved["effective"]["x"], 1)
            self.assertEqual(resolved["sources"]["overrides"], ["x=1"])

    def test_effective_document_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run/effective_config.json"
            write_effective_config(path, {"format_version": 1, "effective": {"x": 1}})
            self.assertEqual(json.loads(path.read_text())["effective"]["x"], 1)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_unknown_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"format_version": 99}')
            with self.assertRaises(ValueError):
                resolve_config(path)


if __name__ == "__main__":
    unittest.main()
