import contextlib
import io
import os
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
                env=cli._tk_environment(),
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

    def test_windows_tk_paths_use_the_selected_base_installation(self):
        import _tkinter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tcl"
            tcl_dir = root / f"tcl{_tkinter.TCL_VERSION}"
            tk_dir = root / f"tk{_tkinter.TK_VERSION}"
            tcl_dir.mkdir(parents=True)
            tk_dir.mkdir()
            (tcl_dir / "init.tcl").touch()
            (tk_dir / "tk.tcl").touch()
            with patch.object(sys, "platform", "win32"), patch.object(sys, "base_prefix", directory):
                with patch.dict(os.environ, {}, clear=True):
                    environment = cli._tk_environment()
                    self.assertEqual(environment["TCL_LIBRARY"], str(tcl_dir))
                    self.assertEqual(environment["TK_LIBRARY"], str(tk_dir))
                    self.assertNotIn("TCL_LIBRARY", os.environ)
                with patch.dict(os.environ, {"TCL_LIBRARY": "custom-tcl", "TK_LIBRARY": "custom-tk"}):
                    environment = cli._tk_environment()
                    self.assertEqual(environment["TCL_LIBRARY"], "custom-tcl")
                    self.assertEqual(environment["TK_LIBRARY"], "custom-tk")

    def test_missing_tk_runtime_is_not_replaced_with_an_unrelated_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "platform", "win32"), patch.object(sys, "base_prefix", directory):
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(cli._tk_environment(), {})


if __name__ == "__main__":
    unittest.main()