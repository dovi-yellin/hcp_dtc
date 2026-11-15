"""Numerical provenance and portable CLI checks for the simulation-only S3 replay."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scripts import make_figure_S3 as s3


class FigureS3Tests(unittest.TestCase):
    def test_verified_mean_sem_and_fit(self):
        data = s3.load_data()
        self.assertEqual(data["realization_means"].shape, (14, 15))
        self.assertEqual(int(data["take"].sum()), 8)
        # Archived values independently checked against all 14 x 4 window summaries.
        np.testing.assert_allclose(data["means"][[0, -1]],
                                  [.31095465991906474, .6445880192013579], atol=1e-14)
        np.testing.assert_allclose(data["sem"][[0, -1]],
                                  [.0004334745656839322, .008715357826440295], atol=1e-14)
        self.assertAlmostEqual(data["provenance"]["expected_fit"]["alpha"], .24009661045865868)

    def test_checksum_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            shutil.copytree(s3.DATA, folder / "data")
            with (folder / "data/tau120_network_means.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                s3.load_data(folder / "data")

    def test_duplicate_table_rejected_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            shutil.copytree(s3.DATA, folder / "data")
            source = folder / "data/tau120_network_means.csv"
            lines = source.read_text().splitlines()
            source.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
            provenance = folder / "data/provenance.json"
            info = json.loads(provenance.read_text())
            info["csv_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            provenance.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                s3.load_data(folder / "data")

    def test_plotted_fit_and_error_bars_use_same_data(self):
        data = s3.load_data()
        fig = s3.draw(data)
        try:
            for index, ax in enumerate(fig.axes):
                fitted = ax.containers[-1]
                np.testing.assert_array_equal(fitted.lines[0].get_xdata(), (1-data["margins"])[data["take"]])
                np.testing.assert_array_equal(fitted.lines[0].get_ydata(), data["means"][data["take"]])
                segments = fitted.lines[2][0].get_segments()
                np.testing.assert_allclose([s[1, 1]-s[0, 1] for s in segments],
                                          2*data["sem"][data["take"]], atol=1e-14)
                line = next(line for line in ax.lines if line.get_linestyle() == "-")
                gain = line.get_xdata()
                expected = .1927971199265775 * (1-gain)**(-.24009661045865868)
                np.testing.assert_array_equal(line.get_ydata(), expected)
                self.assertEqual(sum(line.get_linestyle() == ":" for line in ax.lines), 2 if index else 0)
        finally:
            s3.plt.close(fig)

    def test_caption_matches_reviewed_documentation(self):
        caption = s3.make_caption(s3.load_data())
        self.assertEqual(caption, (s3.ROOT / "docs/figure_S3_caption.md").read_text(encoding="utf-8"))

    def test_standalone_replay_without_repository_or_hcp_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "scripts").mkdir()
            shutil.copyfile(s3.__file__, folder / "scripts/make_figure_S3.py")
            shutil.copytree(s3.DATA, folder / "data/figure_S3")
            output = folder / "rendered"
            result = subprocess.run(
                [sys.executable, str(folder / "scripts/make_figure_S3.py"),
                 "--output-dir", str(output)], cwd=temp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ["figure_S3.png", "figure_S3.pdf", "figure_S3.svg", "caption.md", "source_data.csv"]:
                self.assertGreater((output / name).stat().st_size, 0)
            record = json.loads((output / "verification.json").read_text())
            for name, digest in record["files_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertTrue(record["one_fit_shared_by_both_panels"])


if __name__ == "__main__":
    unittest.main()
