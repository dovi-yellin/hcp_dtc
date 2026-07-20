# ============================================================
# Figure 5 - regional BOLD power spectra and their low-dimensional structure.
#
#   A : regional PSD family (z-scored spectra), coloured by fALFF-based DTC
#   B : dimensionality of regional spectral shape (SVD of the log-PSDs) + the
#       two leading shape modes (inset)
#   C : spectral alignment by removal of the dominant tilt (PC1) mode
#   D : two-mode Lorentzian approximation of the regional PSDs
#
# Inputs (relative to the repository root; see data/README.md):
#   data/all_subjects_all_rois_mean_psd_v2.pickle             z-scored regional PSDs (A, C, D)
#   data/all_subjects_all_rois_psd_tensor_raw_centered.pickle demean regional PSDs (B, C)
#       -> produced by scripts/build_regional_psd_tensor.py
#   data/criticality_markers_subject_n*.pickle                fALFF -> control parameter
#
# Run from the repository root:  python scripts/make_figure_5.py
# ============================================================

import os
import pickle
from os.path import join
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

from shape_collapse_helpers import fit_two_mode_psd_array, fit_two_mode_psd_array_joint

# ---------------- config ----------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
path_str = str(_REPO_ROOT / "data")
fig_dir = str(_REPO_ROOT / "figures")
os.makedirs(fig_dir, exist_ok=True)

num_subjects = 40
num_rois = 180
lettersize = 20
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # editable text

# Two-mode Lorentzian fit feeding panel D:
#   "per_region" -> fit_two_mode_psd_array       (each ROI fit independently)
#   "joint"      -> fit_two_mode_psd_array_joint (shared A / w / f_fast)
D_FIT = "per_region"

# ---------------- data: z-scored regional PSDs for A / C / D ----------------
with open(join(path_str, "all_subjects_all_rois_mean_psd_v2.pickle"), "rb") as h:
    empirical_psd = np.array(pickle.load(h)).T          # (73 freqs x 180 ROIs)

df = 1 / 300
fr = np.arange(0, 0.5 + df / 2, df)
fr_tag = fr[(fr >= 0.01) & (fr <= 0.25)]

# fALFF -> control parameter (DTC axis used for the jet colouring)
mean_fALFF_arr = []
for sub_ind in range(num_subjects):
    with open(join(path_str, f"criticality_markers_subject_n{sub_ind + 1}.pickle"), "rb") as h:
        mean_fALFF_arr.append(pickle.load(h)["mean_falff"])
falff = np.array(mean_fALFF_arr)[:, 0:180].mean(axis=0)

offset, exponent_p = 0.2, 0.24
control_param = 1 - (offset / falff) ** (1 / exponent_p)

norm = Normalize(vmin=np.min(control_param), vmax=np.max(control_param))
cmap = get_cmap("jet")
colors = [cmap(norm(cp)) for cp in control_param]

# two-mode Lorentzian approximation for panel D
if D_FIT == "joint":
    joint_fit = fit_two_mode_psd_array_joint(freqs=fr_tag, psd_arr=empirical_psd.T,
                                             fit_band=(0.01, 0.5))
    fitted_psd_arr = np.vstack([joint_fit["psd_fit_arr"]])
    print("[D] joint fit  global log-R2 =", joint_fit["r2_log_global"])
else:
    fits = fit_two_mode_psd_array(fr_tag, empirical_psd.T, fit_band=(0.01, 0.5))
    fitted_psd_arr = np.vstack([f["psd_fit"] for f in fits])
    r2 = np.array([f["r2_log"] for f in fits])
    print(f"[D] per-region two-mode Lorentzian: median log-R2 {np.median(r2):.3f}, "
          f"min {r2.min():.3f}, %>0.9 = {100 * np.mean(r2 > 0.9):.0f}%")

# ---------------- data: demean regional PSDs for B (and the tilt mode for C) ----------------
with open(join(path_str, "all_subjects_all_rois_psd_tensor_raw_centered.pickle"), "rb") as h:
    d = pickle.load(h)
Pg = np.asarray(d["psd_arr"]).mean(axis=0)              # (180 ROIs x 73 freqs)
fr_b = np.asarray(d["fr_tag"])

xb = np.log10(fr_b)
Yb = np.log10(Pg)
X1 = np.vstack([np.ones_like(xb), xb]).T
chi = np.array([-np.linalg.lstsq(X1, Yb[r], rcond=None)[0][1] for r in range(Yb.shape[0])])

Ms = Yb - Yb.mean(axis=1, keepdims=True)               # remove each ROI's overall power
Dv = Ms - Ms.mean(axis=0, keepdims=True)               # deviations from the mean shape
U, S, Vt = np.linalg.svd(Dv, full_matrices=False)
varexp = 100 * S ** 2 / np.sum(S ** 2)
scores = U * S
pc1_score, pc1_mode, pc2_mode = scores[:, 0], Vt[0], Vt[1]
if np.corrcoef(pc1_score, chi)[0, 1] < 0:              # orient PC1 to increase with exponent
    pc1_score, pc1_mode = -pc1_score, -pc1_mode
r_pc1_chi = np.corrcoef(pc1_score, chi)[0, 1]

sgn = np.sign(pc1_mode)
ic = np.where(np.diff(sgn) != 0)[0][0]
f0 = fr_b[ic] - pc1_mode[ic] * (fr_b[ic + 1] - fr_b[ic]) / (pc1_mode[ic + 1] - pc1_mode[ic])
print(f"[B] PC1 {varexp[0]:.1f}%  PC2 {varexp[1]:.1f}%  | PC1 vs chi r = {r_pc1_chi:.3f} "
      f"| PC1 zero-crossing = {f0:.4f} Hz")


# ============================================================
# Panel C: alignment of the regional spectra by removal of the dominant tilt.
# The fixed PC1 mode is learned from the demean spectra (panel B) and removed
# from the z-scored spectra shown in panel A. No crossing point, fALFF, or
# fitted rescaling parameter is used.
# ============================================================
def align_psds_by_fixed_mode(psd_freq_by_roi, psd_freqs, mode, mode_freqs):
    """Remove the contribution of a fixed spectral mode from regional log-PSDs.

    Parameters
    ----------
    psd_freq_by_roi : array (n_freqs, n_rois), strictly positive PSDs.
    psd_freqs       : array (n_freqs,), their frequencies.
    mode            : array, the fixed PC1 loading (from panel B).
    mode_freqs      : array, frequencies corresponding to the fixed mode.

    Returns
    -------
    aligned_psd (n_freqs, n_rois), geometric-mean (n_freqs,),
    projection scores (n_rois,), fraction of shape variance removed, mode on grid.
    """
    psd_freq_by_roi = np.asarray(psd_freq_by_roi, dtype=float)
    psd_freqs = np.asarray(psd_freqs, dtype=float)
    mode = np.asarray(mode, dtype=float)
    mode_freqs = np.asarray(mode_freqs, dtype=float)

    if np.any(~np.isfinite(psd_freq_by_roi)) or np.any(psd_freq_by_roi <= 0):
        raise ValueError("All PSD values must be finite and strictly positive.")

    # transfer the mode to the PSD frequency grid if needed
    if len(psd_freqs) != len(mode_freqs) or not np.allclose(psd_freqs, mode_freqs):
        mode_target = np.interp(np.log10(psd_freqs), np.log10(mode_freqs), mode)
    else:
        mode_target = mode.copy()

    # centre and unit-normalise the mode after any interpolation
    mode_target -= mode_target.mean()
    mode_norm = np.linalg.norm(mode_target)
    if mode_norm == 0:
        raise ValueError("The supplied PC1 mode has zero norm.")
    mode_target /= mode_norm

    log_psd = np.log10(psd_freq_by_roi.T)                       # ROI x freq
    roi_offsets = log_psd.mean(axis=1, keepdims=True)           # per-ROI overall power
    regional_shapes = log_psd - roi_offsets
    mean_shape = regional_shapes.mean(axis=0, keepdims=True)
    shape_deviations = regional_shapes - mean_shape

    projection_scores = shape_deviations @ mode_target
    pc1_contribution = np.outer(projection_scores, mode_target)
    residual_shapes = shape_deviations - pc1_contribution      # remove the tilt only
    aligned_shapes = mean_shape + residual_shapes

    # give every aligned curve a common overall log-power offset
    common_offset = float(roi_offsets.mean())
    aligned_log_psd = aligned_shapes + common_offset

    variance_removed = 1.0 - np.sum(residual_shapes ** 2) / np.sum(shape_deviations ** 2)
    return ((10.0 ** aligned_log_psd).T, 10.0 ** aligned_log_psd.mean(axis=0),
            projection_scores, variance_removed, mode_target)


psd_pc1_aligned, psd_pc1_aligned_mean, _, pc1_variance_removed, _ = align_psds_by_fixed_mode(
    psd_freq_by_roi=empirical_psd, psd_freqs=fr_tag, mode=pc1_mode, mode_freqs=fr_b)
print(f"[C] Fixed PC1 mode removes {100 * pc1_variance_removed:.1f}% of shape variance")

PC1_COLOR, PC2_COLOR, GREY = cmap(0.12), cmap(0.88), "0.55"

# ============================================================
# figure: 2 rows x 2 cols
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# ---- A: regional PSD family ----
lines = axA.loglog(fr_tag, empirical_psd)
for ln, c in zip(lines, colors):
    ln.set_color(c)
axA.set_xlabel("Freq (Hz)", fontsize=lettersize)
axA.set_ylabel("Power (Z-score)", fontsize=lettersize)
axA.tick_params(axis="both", labelsize=lettersize)
axA.set_ylim(0.8, 11)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = axA.inset_axes([0.76, 0.55, 0.03, 0.34])
cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
cbar.set_label("Control parameter G", fontsize=lettersize - 4)
cbar.ax.tick_params(labelsize=lettersize - 5, length=3)

# ---- B: shape dimensionality (SVD of the demean spectra) ----
k = 5
bars = axB.bar(np.arange(1, k + 1), varexp[:k], width=0.6, color=PC1_COLOR, zorder=3)
bars[1].set_color(PC2_COLOR)
axB.text(1, varexp[0] + 2.5, f"{varexp[0]:.1f}%", ha="center",
         fontsize=lettersize - 2, color=PC1_COLOR)
axB.text(2, varexp[1] + 2.5, f"{varexp[1]:.1f}%", ha="center",
         fontsize=lettersize - 2, color=PC2_COLOR)
axB.set_xticks(np.arange(1, k + 1))
axB.set_xlabel("Shape component", fontsize=lettersize)
axB.set_ylabel("Variance explained (%)", fontsize=lettersize)
axB.tick_params(axis="both", labelsize=lettersize)
axB.set_ylim(0, 110)

ins = axB.inset_axes([0.40, 0.47, 0.58, 0.50])
ins.plot(fr_b, pc1_mode, color=PC1_COLOR, lw=4.0, label="PC1", zorder=3)
ins.plot(fr_b, pc2_mode, color=PC2_COLOR, lw=4.0, label="PC2", zorder=3)
ins.axhline(0, color=GREY, lw=1.0, zorder=1)
ins.axvline(f0, color="k", ls="--", lw=1.4, zorder=2)
ins.set_xscale("log")
ins.set_xlabel("Freq (Hz)", fontsize=lettersize, labelpad=2)
ins.set_ylabel("Mode loading", fontsize=lettersize, labelpad=2)
ins.tick_params(axis="both", labelsize=lettersize, pad=2, length=4)
ins.legend(frameon=False, fontsize=lettersize, loc="upper right",
           handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)

# ---- C: tilt-removal spectral alignment ----
lines = axC.loglog(fr_tag, psd_pc1_aligned)
for ln, c in zip(lines, colors):
    ln.set_color(c)
axC.loglog(fr_tag, psd_pc1_aligned_mean, linewidth=5, color="k", zorder=5)  # geometric mean
axC.set_xlabel("Freq (Hz)", fontsize=lettersize)
axC.set_ylabel("Power (Z-score)", fontsize=lettersize)
axC.tick_params(axis="both", labelsize=lettersize)
axC.set_ylim(0.8, 11)

# ---- D: two-mode Lorentzian approximation ----
for i in range(empirical_psd.shape[1]):
    axD.loglog(fr_tag, empirical_psd[:, i], color=colors[i], alpha=0.18, lw=1.0)
for i in range(fitted_psd_arr.shape[0]):
    axD.loglog(fr_tag, fitted_psd_arr[i], "--", color=colors[i], alpha=0.95, lw=2.2)
axD.set_xlabel("Freq (Hz)", fontsize=lettersize)
axD.set_ylabel("Power (Z-score)", fontsize=lettersize)
axD.tick_params(axis="both", labelsize=lettersize)
axD.set_ylim(0.8, 11)

# ---- panel letters ----
for a, lab in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
    a.annotate(lab, xy=(-0.12, 1.04), xycoords="axes fraction",
               fontsize=lettersize + 6, fontweight="bold", va="top")

fig.tight_layout()
fig.savefig(join(fig_dir, "figure_5.pdf"), bbox_inches="tight")
fig.savefig(join(fig_dir, "figure_5.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nWrote ->", join(fig_dir, "figure_5.[pdf|png]"))
