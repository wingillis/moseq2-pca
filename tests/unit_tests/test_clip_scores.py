import os
import tempfile
from pathlib import Path
from unittest import TestCase

import h5py
import numpy as np

from moseq2_pca.helpers.wrappers import clip_scores_wrapper


class TestClipScores(TestCase):

    def _make_scores_file(self, nframes=20, npcs=3):
        tmpdir = tempfile.mkdtemp()
        pca_file = Path(tmpdir) / "pca_scores.h5"
        rng = np.random.default_rng(0)
        with h5py.File(pca_file, "w") as f:
            f.create_dataset("metadata/fps", data=30.0)
            f.create_dataset(
                f"scores/some-uuid", data=rng.standard_normal((nframes, npcs)).astype("float32")
            )
            f.create_dataset(
                f"scores_idx/some-uuid",
                data=np.arange(nframes, dtype="float32"),
            )
        return pca_file

    def test_clip_from_beginning(self):
        pca_file = self._make_scores_file()
        out = clip_scores_wrapper(pca_file, 5)
        assert out.exists()
        with h5py.File(pca_file) as f, h5py.File(out) as g:
            assert g["scores/some-uuid"].shape == (15, 3)
            assert g["scores_idx/some-uuid"].shape == (15,)
            np.testing.assert_allclose(g["scores/some-uuid"][()], f["scores/some-uuid"][5:])
            np.testing.assert_allclose(
                g["scores_idx/some-uuid"][()], f["scores_idx/some-uuid"][5:]
            )
            assert "metadata" in g

    def test_clip_from_end(self):
        pca_file = self._make_scores_file()
        out = clip_scores_wrapper(pca_file, 5, from_end=True)
        with h5py.File(pca_file) as f, h5py.File(out) as g:
            assert g["scores/some-uuid"].shape == (15, 3)
            np.testing.assert_allclose(g["scores/some-uuid"][()], f["scores/some-uuid"][:-5])
            np.testing.assert_allclose(
                g["scores_idx/some-uuid"][()], f["scores_idx/some-uuid"][:-5]
            )
