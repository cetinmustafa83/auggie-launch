#!/usr/bin/env python3
"""Smoke tests for install.sh.

The installer is the first thing a user runs, and none of its seven real bugs
(found by inspection) were covered by anything. These tests install into a
throwaway prefix and assert on the observable result, so a regression shows up
here rather than on someone's machine.

Skipped automatically when bash is unavailable.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
INSTALLER = os.path.join(REPO, "install.sh")


def has_bash() -> bool:
    return shutil.which("bash") is not None


def run_installer(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", INSTALLER, *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )


@unittest.skipUnless(has_bash(), "bash is required")
class TestInstaller(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="auggie-launch-install-")
        self.bin_dir = os.path.join(self.tmp, "bin")
        self.config_dir = os.path.join(self.tmp, "cfg")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, *extra):
        return run_installer(
            "--bin-dir", self.bin_dir,
            "--config-dir", self.config_dir,
            "--skip-cli", "--skip-verify",
            *extra,
        )

    def test_syntax_is_valid(self):
        result = subprocess.run(["bash", "-n", INSTALLER], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bin_dir_is_honoured(self):
        """A regression here writes into the real ~/.local/bin instead."""
        result = self._install()
        self.assertEqual(result.returncode, 0, result.stderr)
        wrapper = os.path.join(self.bin_dir, "auggie-launch")
        self.assertTrue(os.path.isfile(wrapper), f"wrapper not at {wrapper}\n{result.stdout}")
        self.assertTrue(os.access(wrapper, os.X_OK), "wrapper is not executable")
        home_wrapper = os.path.join(os.path.expanduser("~"), ".local", "bin", "auggie-launch")
        self.assertNotIn(f"installed {home_wrapper}", result.stdout)

    def test_config_dir_is_honoured(self):
        result = self._install("--force-env")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.isfile(os.path.join(self.config_dir, ".env")), result.stdout)

    def test_env_is_created_private(self):
        self._install("--force-env")
        env_path = os.path.join(self.config_dir, ".env")
        mode = os.stat(env_path).st_mode & 0o777
        self.assertEqual(mode, 0o600, f"expected 600, got {mode:o}")

    def test_no_env_skips_creation_and_says_so(self):
        result = self._install("--no-env")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.config_dir, ".env")))
        # It must not invite the user to edit a file it never wrote.
        self.assertIn("skipped", result.stdout)

    def test_existing_env_is_kept_without_force(self):
        first = self._install("--force-env")
        self.assertEqual(first.returncode, 0, first.stderr)
        env_path = os.path.join(self.config_dir, ".env")
        with open(env_path, "a", encoding="utf-8") as fh:
            fh.write("\n# marker\n")
        second = self._install()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("keep existing config", second.stdout)
        with open(env_path, encoding="utf-8") as fh:
            self.assertIn("# marker", fh.read())

    def test_link_mode_symlinks_the_entrypoint(self):
        result = self._install("--link")
        self.assertEqual(result.returncode, 0, result.stderr)
        wrapper = os.path.join(self.bin_dir, "auggie-launch")
        self.assertTrue(os.path.islink(wrapper), result.stdout)

    def test_wrapper_runs_the_package(self):
        self._install()
        wrapper = os.path.join(self.bin_dir, "auggie-launch")
        result = subprocess.run([wrapper, "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("auggie-launch", result.stdout)

    def test_installed_wrapper_reports_environment(self):
        self._install("--force-env")
        wrapper = os.path.join(self.bin_dir, "auggie-launch")
        result = subprocess.run([wrapper, "--print-env"], capture_output=True, text=True, timeout=60)
        # With an unconfigured env it may exit non-zero, but it must still print
        # the resolved shape rather than traceback.
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("AUGGIE_LAUNCH_BASE_URL", result.stdout + result.stderr)

    def test_unknown_option_is_rejected(self):
        result = run_installer("--not-a-flag")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown option", result.stderr)

    def test_help_lists_real_flags(self):
        result = subprocess.run(["bash", INSTALLER, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        for flag in ("--bin-dir", "--config-dir", "--force-env", "--no-env",
                     "--skip-cli", "--skip-verify", "--link", "--pipx"):
            self.assertIn(flag, result.stdout, flag)
        # The removed 9router flag must not linger in the help text.
        self.assertNotIn("9router", result.stdout)


if __name__ == "__main__":
    unittest.main()
