import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from polarcam import cli


class LiveLauncherTests(unittest.TestCase):
    def test_help_does_not_start_the_application(self):
        with patch("polarcam.cli.subprocess.run") as launch:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as exit_info:
                    cli.main(["--help"])

        self.assertEqual(exit_info.exception.code, 0)
        self.assertIn("--data-dir", output.getvalue())
        launch.assert_not_called()

    def test_launch_preserves_interpreter_and_script_layout(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir) / "recording output"
            with patch("polarcam.cli.subprocess.run") as launch:
                launch.return_value.returncode = 0
                result = cli.main(["--data-dir", str(data_dir)])

            application = Path(cli.__file__).resolve().parent / "live" / "Spinners_gui_live.py"
            self.assertTrue(application.is_file())
            self.assertTrue((application.parent / "fetch_frames.py").is_file())
            self.assertTrue(data_dir.is_dir())
            launch.assert_called_once_with(
                [sys.executable, str(application)],
                cwd=data_dir.resolve(),
                check=False,
            )
            self.assertEqual(result, 0)

    def test_default_directory_and_child_exit_status(self):
        with patch("polarcam.cli.subprocess.run") as launch:
            launch.return_value.returncode = 7
            result = cli.main([])

        self.assertEqual(launch.call_args.kwargs["cwd"], Path.cwd().resolve())
        self.assertEqual(result, 7)

    def test_module_entry_point_propagates_exit_status(self):
        with patch("polarcam.cli.main", return_value=9):
            with self.assertRaises(SystemExit) as exit_info:
                runpy.run_module("polarcam", run_name="__main__")

        self.assertEqual(exit_info.exception.code, 9)


if __name__ == "__main__":
    unittest.main()