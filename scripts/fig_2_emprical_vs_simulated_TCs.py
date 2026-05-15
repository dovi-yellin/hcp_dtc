import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import HCP_fcd_utils as utils
from initParams import initParams
from models import RateModel


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "results" / "figures"
FIGURE_PATH = FIGURE_DIR / "fig_2_empirical_vs_simulated_TCs.png"

SELECTED_SUBJECT = 7
ROI_CONFIGS = [
    ("Low fALFF", 9, 0.080),
    ("Med fALFF", 1, 0.096),
    ("High fALFF", 2, 0.099),
]

N_UNITS = 400
TOTAL_TIME_SEC = 500
PLOT_DURATION_SEC = 300
TRACE_RANGE = slice(0, 50)
RANDOM_SEED = 53


def z_score_columns(activity):
    mean = activity.mean(axis=0, keepdims=True)
    std = activity.std(axis=0, keepdims=True)
    return np.divide(activity - mean, std, out=np.zeros_like(activity), where=std != 0)


def load_empirical_roi_tcs(subject_id, roi_configs):
    return [
        utils.get_subject_rois_tc(subject_id, roi_index, z_score=True)
        for _, roi_index, _ in roi_configs
    ]


def simulate_roi_tcs(roi_configs):
    np.random.seed(RANDOM_SEED)

    params = initParams(N=N_UNITS, total_time=TOTAL_TIME_SEC)
    params.tau = 20.0

    simulated_roi_tcs = []
    for _, _, gamma in roi_configs:
        params.gamma = gamma
        print("Model params are: ", params)
        model = RateModel(params)

        activity = utils.run_model_iteration(
            model,
            params,
            z_score=False,
            hrf_convole=True,
        )
        simulated_tc = activity[:PLOT_DURATION_SEC, :]
        simulated_roi_tcs.append(z_score_columns(simulated_tc))

    return simulated_roi_tcs


def plot_tcs(empirical_roi_tcs, simulated_roi_tcs, roi_configs, output_path):
    plt.rcParams.update({"font.size": 18})
    fig, axs = plt.subplots(
        len(roi_configs),
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )
    plt.setp(axs, yticks=[-2.5, 0, 2.5])

    for row, (label, _, gamma) in enumerate(roi_configs):
        axs[row, 0].plot(empirical_roi_tcs[row][:PLOT_DURATION_SEC, TRACE_RANGE])
        axs[row, 1].plot(simulated_roi_tcs[row][:, TRACE_RANGE])

        axs[row, 0].text(
            0.02,
            0.92,
            label,
            transform=axs[row, 0].transAxes,
            va="top",
            fontsize=14,
        )
        axs[row, 1].text(
            0.02,
            0.92,
            f"gamma = {gamma:.3f}",
            transform=axs[row, 1].transAxes,
            va="top",
            fontsize=14,
        )

    axs[0, 0].set_title("Empirical HCP", fontsize=18)
    axs[0, 1].set_title("Simulated", fontsize=18)
    axs[-1, 0].set_xlabel("Time (s)", fontsize=24)
    axs[-1, 1].set_xlabel("Time (s)", fontsize=24)
    fig.text(
        0.01,
        0.5,
        "Activity (z-score)",
        va="center",
        rotation="vertical",
        fontsize=24,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, dpi=300)
    plt.close(fig)


def main():
    os.chdir(REPO_ROOT)
    empirical_roi_tcs = load_empirical_roi_tcs(SELECTED_SUBJECT, ROI_CONFIGS)
    simulated_roi_tcs = simulate_roi_tcs(ROI_CONFIGS)
    plot_tcs(empirical_roi_tcs, simulated_roi_tcs, ROI_CONFIGS, FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
