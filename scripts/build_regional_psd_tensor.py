# ============================================================
# Build the (demean, non-normalised) regional power-spectral-density tensor
# consumed by make_figure_5.py (panels B, C) and make_figure_S5.py (panel A).
#
# For each subject and cortical ROI, the per-voxel BOLD time courses are
# demeaned (not z-scored), a multitaper PSD is computed per voxel and averaged
# within the ROI, and the spectrum is restricted to 0.01-0.25 Hz.
#
# Input : data/hcp_subject_n*.pickle   (per-subject voxel time courses; see data/README.md)
# Output: data/all_subjects_all_rois_psd_tensor_raw_centered.pickle
#           {"psd_arr": (n_subjects x n_rois x n_freq), "fr_tag": frequencies}
#
# Run once from the repository root:  python scripts/build_regional_psd_tensor.py
# ============================================================

import pickle
from os.path import join
from pathlib import Path

import numpy as np
import HCP_fcd_utils as utils

_REPO_ROOT = Path(__file__).resolve().parents[1]
path_str = str(_REPO_ROOT / "data")
utils.path_str = path_str            # make the loader independent of the working directory

NUM_SUBJECTS = 40
NUM_ROIS = 180
FMIN, FMAX = 0.01, 0.25
OUT = join(path_str, "all_subjects_all_rois_psd_tensor_raw_centered.pickle")


def main():
    all_psd = []
    fr_tag = None
    for s in range(NUM_SUBJECTS):
        roi_psds = []
        for r in range(NUM_ROIS):
            # subject numbers are 1-based; demean each voxel (no z-scoring)
            roi_tc = utils.get_subject_rois_tc(s + 1, r, z_score=False, demean=True)
            mean_s, fr = utils.compute_mean_psd(roi_tc)
            mask = (fr >= FMIN) & (fr <= FMAX)
            if fr_tag is None:
                fr_tag = fr[mask]
            roi_psds.append(mean_s[mask])
        all_psd.append(roi_psds)
        print(f"  subject {s + 1}/{NUM_SUBJECTS} done")

    psd_arr = np.asarray(all_psd)                # (subjects x ROIs x freq)
    with open(OUT, "wb") as h:
        pickle.dump({"psd_arr": psd_arr, "fr_tag": fr_tag,
                     "num_subjects": NUM_SUBJECTS, "num_rois": NUM_ROIS,
                     "z_score": False, "demean": True}, h)
    print("wrote", OUT, psd_arr.shape)


if __name__ == "__main__":
    main()
