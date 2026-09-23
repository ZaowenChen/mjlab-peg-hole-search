import tempfile
import unittest
from pathlib import Path

import torch

from mjlab_contact_prep.data.state_bank import (
    STATE_BANK_FORMAT_VERSION,
    load_state_bank,
    save_state_bank,
)


def legacy_bank():
    return {
        "physics": {"qpos": torch.zeros(1, 2)},
        "controller": {"state": torch.zeros(1)},
        "term": {"command": torch.zeros(1, 6)},
        "probe": {"phase": torch.zeros(1)},
        "rows": [{"eligible": True}],
    }


class StateBankFormatTests(unittest.TestCase):
    def test_legacy_bank_upgrades_without_rewrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            torch.save(legacy_bank(), path)
            before = path.read_bytes()
            loaded = load_state_bank(path)
            self.assertEqual(loaded["format_version"], STATE_BANK_FORMAT_VERSION)
            self.assertEqual(loaded["format_metadata"]["source_format_version"], 0)
            self.assertEqual(path.read_bytes(), before)

    def test_new_bank_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "versioned.pt"
            save_state_bank(path, legacy_bank())
            loaded = load_state_bank(path)
            self.assertEqual(loaded["format_metadata"]["source_format_version"], 1)

    def test_missing_full_state_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pt"
            bad = legacy_bank()
            del bad["controller"]
            torch.save(bad, path)
            with self.assertRaises(ValueError):
                load_state_bank(path)


if __name__ == "__main__":
    unittest.main()
