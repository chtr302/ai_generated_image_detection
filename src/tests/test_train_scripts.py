from __future__ import annotations

import unittest
from pathlib import Path


class TrainScriptTest(unittest.TestCase):
    def test_powershell_scripts_use_data_root(self) -> None:
        for script in ("train_single_gpu.ps1", "train_dual_t4_ddp.ps1"):
            text = Path("scripts", script).read_text(encoding="utf-8")
            self.assertIn("$DataRoot", text)
            self.assertIn("--data-root", text)
            self.assertNotIn("--train-dir", text)

    def test_bash_script_uses_data_root(self) -> None:
        text = Path("scripts", "train_dual_t4_ddp.sh").read_text(encoding="utf-8")
        self.assertIn("DATA_ROOT", text)
        self.assertIn("--data-root", text)
        self.assertNotIn("TRAIN_DIR", text)


if __name__ == "__main__":
    unittest.main()
