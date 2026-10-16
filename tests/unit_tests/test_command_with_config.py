"""
Unit tests for the restored release-era config-file behavior:

- ``--config-file`` is a subcommand-level option on train-pca / apply-pca /
  compute-changepoints and reads a FLAT yaml file (option names at the top
  level, no ``pca:`` section).
- Parameter precedence: CLI options > config file values > click defaults.
- ``combine_new_config`` merges flat keys and filters output paths.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

import moseq2_pca.cli as cli_module
from moseq2_pca.cli import cli, train_pca
from moseq2_pca.util import combine_new_config, read_yaml


class TestCommandWithConfig(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.captured = {}

        def fake_train_pca_wrapper(input_dir, cli_args, output_dir, output_file):
            self.captured.update(cli_args)
            return {
                "pca_file": "/tmp/pca.h5",
                "output_dir": "filtered",
                "keep_me": 1,
            }

        self._orig_wrapper = cli_module.train_pca_wrapper
        cli_module.train_pca_wrapper = fake_train_pca_wrapper
        self.addCleanup(
            setattr, cli_module, "train_pca_wrapper", self._orig_wrapper
        )

    def invoke_train_pca(self, extra_args):
        return self.runner.invoke(
            train_pca, ["-i", self.tmp.name] + extra_args, catch_exceptions=False
        )

    def test_train_pca_has_config_file_option(self):
        result = self.runner.invoke(train_pca, ["--help"])
        self.assertIn("--config-file", result.output)

    def test_group_has_no_config_file_option(self):
        result = self.runner.invoke(cli, ["--help"])
        self.assertNotIn("--config-file", result.output.split("Commands")[0])

    def test_defaults_without_config(self):
        result = self.invoke_train_pca([])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.captured["chunk_size"], 4000)

    def test_flat_config_overrides_default(self):
        config_file = Path(self.tmp.name) / "config.yaml"
        config_file.write_text(
            "chunk_size: 1234\ngaussfilter_space: [2.25, 1.5]\n"
            "output_dir: /should/be/ignored\n"
        )

        result = self.invoke_train_pca(["--config-file", str(config_file)])
        self.assertEqual(result.exit_code, 0)

        self.assertEqual(self.captured["chunk_size"], 1234)
        self.assertEqual(tuple(self.captured["gaussfilter_space"]), (2.25, 1.5))

    def test_cli_overrides_config(self):
        config_file = Path(self.tmp.name) / "config.yaml"
        config_file.write_text("chunk_size: 1234\n")

        result = self.invoke_train_pca(
            ["--config-file", str(config_file), "--chunk-size", "555"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.captured["chunk_size"], 555)

    def test_sectioned_config_is_not_supported(self):
        # the new-style sectioned config has no effect on flat options
        config_file = Path(self.tmp.name) / "sectioned.yaml"
        config_file.write_text("pca:\n  chunk_size: 99\n")

        result = self.invoke_train_pca(["--config-file", str(config_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.captured["chunk_size"], 4000)

    def test_combine_new_config_writes_back_flat(self):
        config_file = Path(self.tmp.name) / "config.yaml"
        config_file.write_text("chunk_size: 1234\nuser_key: hello\n")

        combine_new_config(
            str(config_file),
            {"pca_file": "/tmp/pca.h5", "output_dir": "filtered", "keep_me": 1},
        )

        data = read_yaml(config_file)
        # flat merge, new keys at the top level
        self.assertEqual(data["chunk_size"], 1234)
        self.assertEqual(data["user_key"], "hello")
        self.assertEqual(data["pca_file"], "/tmp/pca.h5")
        self.assertEqual(data["keep_me"], 1)
        # output paths are filtered so the config can be reused
        self.assertNotIn("output_dir", data)
        self.assertNotIn("output_file", data)
        # no pca section is created
        self.assertNotIn("pca", data)


if __name__ == "__main__":
    unittest.main()
