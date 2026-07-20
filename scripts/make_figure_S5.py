# ============================================================
# Supplementary Figure S5 - analytical and simulated recurrent-network models
# in relation to the empirical regional PSD structure.
#
#   A : two-mode Lorentzian approximation of the regional PSDs, aligned by
#       removal of the dominant tilt (PC1) mode
#   B : empirical pairwise PSD similarity across ROIs (correlation matrix)
#   C : empirical vs simulated PSD correlation as a function of the control
#       parameter G
#   D : simulated PSDs across DTC, normalised to the empirical power scale,
#       overlaid on the empirical family
#
# Inputs (relative to the repository root; see data/ and results/ README):
#   data/all_subjects_all_rois_mean_psd_v2.pickle             z-scored regional PSDs (A, B, D)
#   data/all_subjects_all_rois_psd_tensor_raw_centered.pickle demean regional PSDs (A tilt mode)
#       -> produced by scripts/build_regional_psd_tensor.py
#   data/criticality_markers_subject_n*.pickle                fALFF -> control parameter
#   results/simulation_N400_per_roi_psd_correlation_analysis_all_gammas_latest.csv   (C)
#   results/simulated_N400_near_criticality_psd_gamma_*_multi_realizations.csv        (D)
#
# Run from the repository root:  python scripts/make_figure_S5.py
# ============================================================

import os
import pickle
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

from shape_collapse_helpers import fit_two_mode_psd_array

_REPO_ROOT = Path(__file__).resolve().parents[1]
path_str = str(_REPO_ROOT / "data")
path_result_str = str(_REPO_ROOT / "results")
fig_dir = str(_REPO_ROOT / "figures")
os.makedirs(fig_dir, exist_ok=True)

lettersize = 19
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
cmap = get_cmap("jet")


def align_psds_by_fixed_mode(psd_freq_by_roi, psd_freqs, mode, mode_freqs):
    """Remove a fixed spectral mode (PC1 tilt) from regional log-PSDs.

    psd_freq_by_roi : (n_freqs, n_rois) positive PSDs
    mode / mode_freqs : the fixed PC1 loading and its frequencies
    Returns (aligned (n_freqs,n_rois), geometric-mean, scores, variance_removed).
    """
    psd_freq_by_roi = np.asarray(psd_freq_by_roi, dtype=float)
    if np.any(~np.isfinite(psd_freq_by_roi)) or np.any(psd_freq_by_roi <= 0):
        raise ValueError("All PSD values must be finite and strictly positive.")

    if len(psd_freqs) != len(mode_freqs) or not np.allclose(psd_freqs, mode_freqs):
        mode_t = np.interp(np.log10(psd_freqs), np.log10(mode_freqs), mode)
    else:
        mode_t = np.asarray(mode, dtype=float).copy()
    mode_t -= mode_t.mean()
    mode_t /= np.linalg.norm(mode_t)

    log_psd = np.log10(psd_freq_by_roi.T)                    # ROI x freq
    roi_offsets = log_psd.mean(axis=1, keepdims=True)        # per-ROI overall power
    shapes = log_psd - roi_offsets
    mean_shape = shapes.mean(axis=0, keepdims=True)
    deviations = shapes - mean_shape

    scores = deviations @ mode_t                             # project onto PC1
    residual = deviations - np.outer(scores, mode_t)         # remove the tilt only
    aligned_log = mean_shape + residual + float(roi_offsets.mean())

    var_removed = 1.0 - np.sum(residual ** 2) / np.sum(deviations ** 2)
    return (10.0 ** aligned_log).T, 10.0 ** aligned_log.mean(axis=0), scores, var_removed


# ---------------- empirical PSDs + control parameter ----------------
with open(join(path_str, "all_subjects_all_rois_mean_psd_v2.pickle"), "rb") as h:
    empirical_psd = np.array(pickle.load(h)).T          # (73 freqs x 180 ROIs)

dfr = 1 / 300
fr = np.arange(0, 0.5 + dfr / 2, dfr)
fr_tag = fr[(fr >= 0.01) & (fr <= 0.25)]

mean_fALFF_arr = []
for s in range(40):
    with open(join(path_str, f"criticality_markers_subject_n{s + 1}.pickle"), "rb") as h:
        mean_fALFF_arr.append(pickle.load(h)["mean_falff"])
falff = np.array(mean_fALFF_arr)[:, 0:180].mean(axis=0)
control_param = 1 - (0.2 / falff) ** (1 / 0.24)
norm = Normalize(vmin=np.min(control_param), vmax=np.max(control_param))
colors = [cmap(norm(cp)) for cp in control_param]

# ---------------- panel A: analytical PSDs, tilt-aligned ----------------
fits = fit_two_mode_psd_array(fr_tag, empirical_psd.T, fit_band=(0.01, 0.5))
fitted_psd_arr = np.vstack([f["psd_fit"] for f in fits])          # (180 x 73), positive

# fixed PC1 tilt learned from the demean spectra
with open(join(path_str, "all_subjects_all_rois_psd_tensor_raw_centered.pickle"), "rb") as h:
    dd = pickle.load(h)
Pg_demean = np.asarray(dd["psd_arr"]).mean(axis=0)                # (180 x 73)
fr_b = np.asarray(dd["fr_tag"])
Yb = np.log10(Pg_demean)
Ms = Yb - Yb.mean(axis=1, keepdims=True)                          # remove ROI overall power
Dv = Ms - Ms.mean(axis=0, keepdims=True)                         # deviations from mean shape
_, _, Vt = np.linalg.svd(Dv, full_matrices=False)
pc1_mode = Vt[0]                                                 # sign-invariant for removal

analytical_aligned, analytical_mean, _, var_removed = align_psds_by_fixed_mode(
    fitted_psd_arr.T, fr_tag, pc1_mode, fr_b)
print(f"[A] tilt-aligned analytical spectra; fixed mode removes "
      f"{100 * var_removed:.1f}% of shape variance")

# ---------------- panel B: empirical pairwise PSD correlation ----------------
corr_mat = np.corrcoef(empirical_psd.T)               # (180 x 180)
iu = np.triu_indices_from(corr_mat, k=1)
emp_mean_corr = float(corr_mat[iu].mean())
print(f"[B] empirical mean pairwise PSD corr = {emp_mean_corr:.3f}")

# ---------------- panel C: empirical vs simulated correlation vs G ----------------
dfc = pd.read_csv(join(path_result_str,
                       "simulation_N400_per_roi_psd_correlation_analysis_all_gammas_latest.csv"))
grp = dfc.groupby("gamma")["mean_corr"]
gam = np.sort(dfc["gamma"].unique())
x_G = gam * 0.2 * 49.881
mean_corr = grp.mean().reindex(gam).values
std_corr = grp.std().reindex(gam).values
max_corr = grp.max().reindex(gam).values
print(f"[C] simulated corr vs G: {len(gam)} gamma levels, G {x_G.min():.3f}-{x_G.max():.3f}")

# ---------------- panel D: normalised simulated PSDs vs empirical ----------------
# Neural simulated mean PSDs across DTC, normalised to equal total power (the
# empirical scale). Under the same normalisation applied to the data, the model
# spectra reproduce the crossing and overlap the empirical family.
emp_bandpow = empirical_psd.sum(axis=0).mean()
sim_gammas = [0.096, 0.097, 0.098, 0.099]                     # -> G ~ 0.96-0.99
sim_curves = []
for gval in sim_gammas:
    fn = join(path_result_str,
              f"simulated_N400_near_criticality_psd_gamma_{str(gval).replace('.', '_')}_multi_realizations.csv")
    if not os.path.exists(fn):
        fn = fn.replace("_multi_realizations.csv", "_multi_realizations_20250805.csv")
    p = np.nanmean(pd.read_csv(fn).values, axis=0)[:len(fr_tag)]
    sim_curves.append((gval * 0.2 * 49.881, p / p.sum() * emp_bandpow))

simM = np.array([c for _, c in sim_curves]).T                # freq x G
sim_vertex = fr_tag[int(np.argmin(np.log(simM).std(axis=1)))]
emp_vertex = fr_tag[int(np.argmin(np.log(empirical_psd).std(axis=1)))]
sim_overlap = np.corrcoef(np.log(simM.mean(1)), np.log(empirical_psd.mean(1)))[0, 1]
print(f"[D] normalised sim crossing {sim_vertex:.3f} Hz vs empirical {emp_vertex:.3f} Hz; "
      f"overlap r = {sim_overlap:.3f}")

# ============================================================
# figure 2 x 2
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# ---- A ----
lines = axA.loglog(fr_tag, analytical_aligned)
for ln, c in zip(lines, colors):
    ln.set_color(c)
axA.loglog(fr_tag, analytical_mean, color="k", lw=5, zorder=5)
axA.set_xlabel("Freq (Hz)", fontsize=lettersize)
axA.set_ylabel("Power (Z-score)", fontsize=lettersize)
axA.tick_params(axis="both", labelsize=lettersize)
axA.set_ylim(0.8, 11)
axA.set_title("Analytical PSDs, tilt-aligned", fontsize=lettersize)

# ---- B ----
im = axB.imshow(corr_mat, cmap="jet", vmin=np.percentile(corr_mat, 2), vmax=1.0)
axB.set_xlabel("ROI index", fontsize=lettersize)
axB.set_ylabel("ROI index", fontsize=lettersize)
axB.tick_params(axis="both", labelsize=lettersize)
axB.set_title("Empirical pairwise PSD similarity", fontsize=lettersize)
cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03)
cb.set_label("Pearson r", fontsize=lettersize - 3)
cb.ax.tick_params(labelsize=lettersize - 4)

# ---- C ----
axC.plot(x_G, mean_corr, color="tab:blue", lw=4, label="Mean (simulated)")
axC.fill_between(x_G, mean_corr - std_corr, mean_corr + std_corr, color="tab:blue", alpha=0.25)
axC.plot(x_G, max_corr, color="tab:orange", lw=3, label="Max (simulated)")
axC.axhline(emp_mean_corr, color="green", ls="--", lw=3, label=f"Empirical ({emp_mean_corr:.2f})")
axC.set_xlabel("Control parameter G", fontsize=lettersize)
axC.set_ylabel("PSD correlation (r)", fontsize=lettersize)
axC.tick_params(axis="both", labelsize=lettersize)
axC.legend(fontsize=lettersize - 5, loc="lower right", frameon=False)
axC.set_title("Empirical vs simulated PSD correlation", fontsize=lettersize)

# ---- D ----
for i in range(empirical_psd.shape[1]):                       # empirical family (grey)
    axD.loglog(fr_tag, empirical_psd[:, i], color="0.82", lw=0.5, alpha=0.5, zorder=1)
for gG, curve in sim_curves:                                  # normalised simulated
    axD.loglog(fr_tag, curve, lw=3.5, color=cmap(norm(gG)), label=f"G = {gG:.2f}", zorder=3)
axD.set_ylim(0.8, 11)
axD.set_xlabel("Freq (Hz)", fontsize=lettersize)
axD.set_ylabel("Power (Z-score)", fontsize=lettersize)
axD.tick_params(axis="both", labelsize=lettersize)
axD.legend(fontsize=lettersize - 5, loc="lower left", frameon=False)
axD.set_title("Normalised simulated PSDs vs empirical", fontsize=lettersize)

# ---- panel letters ----
for a, lab in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
    a.annotate(lab, xy=(-0.12, 1.04), xycoords="axes fraction",
               fontsize=lettersize + 6, fontweight="bold", va="top")

fig.tight_layout()
fig.savefig(join(fig_dir, "figure_S5.pdf"), bbox_inches="tight")
fig.savefig(join(fig_dir, "figure_S5.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nWrote ->", join(fig_dir, "figure_S5.[pdf|png]"))
