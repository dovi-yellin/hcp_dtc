"""Reproduce Supplementary Figure S3 from bundled simulation-only summaries.

Run: python scripts/make_figure_S3.py
Requires NumPy and Matplotlib only; no HCP data or private revision checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_S3"
ORANGE = "#D55E00"
GRAY = "#8c9299"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(data_dir=DATA):
    """Verify provenance, restore the realization x gain table, and check the fit."""
    data_dir = Path(data_dir)
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance["csv_file"] != "tau120_network_means.csv":
        raise ValueError("Unexpected simulation summary filename")
    source = data_dir / provenance["csv_file"]
    if sha256(source) != provenance["csv_sha256"]:
        raise ValueError("Simulation summary checksum differs from provenance")
    margins = np.asarray(provenance["margins"], dtype=float)
    n = provenance["realizations"]
    if (n != 14 or provenance["windows_per_realization"] != 4
            or provenance["tau_ms"] != 120 or provenance["N"] != 400
            or provenance["connection_probability"] != 0.2
            or provenance["effective_filter_peak_s"] != 1.3
            or provenance["sampling_rate_hz"] != 1.0
            or provenance["observation_samples"] != 300
            or margins.shape != (15,) or not np.isfinite(margins).all()
            or not np.all(np.diff(margins) < 0)
            or not np.all((margins > 0) & (margins < 1))):
        raise ValueError("Expected the verified tau120, N400, p0.2 ensemble")
    means_by_realization = np.full((n, len(margins)), np.nan)
    seen = set()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["realization_index", "Gprime_actual", "mean_fALFF_over_4_windows"]:
            raise ValueError("Unexpected simulation summary columns")
        for row in reader:
            realization = int(row["realization_index"])
            margin = float(row["Gprime_actual"])
            value = float(row["mean_fALFF_over_4_windows"])
            matches = np.flatnonzero(margins == margin)
            if not 0 <= realization < n or len(matches) != 1:
                raise ValueError("Unknown realization or stability margin")
            key = (realization, int(matches[0]))
            if key in seen:
                raise ValueError("Duplicate realization/gain summary")
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("fALFF summaries must be finite fractional powers")
            seen.add(key)
            means_by_realization[key] = value
    if len(seen) != n * len(margins):
        raise ValueError("Incomplete realization/gain table")
    means = means_by_realization.mean(axis=0)
    sem = means_by_realization.std(axis=0, ddof=1) / np.sqrt(n)
    lo, hi = provenance["fit_margin_interval"]
    take = (margins >= lo - 1e-12) & (margins <= hi + 1e-12)
    reference = provenance["expected_fit"]
    if take.sum() != reference["gain_points"] or take.sum() != 8:
        raise ValueError("Expected eight gains in the finite fitting interval")
    x, y = np.log10(margins[take]), np.log10(means[take])
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    np.testing.assert_allclose(
        [10 ** intercept, -slope, r2],
        [reference["A"], reference["alpha"], reference["log_space_R2"]],
        atol=1e-12, rtol=1e-12,
    )
    np.testing.assert_allclose([slope, intercept], np.polyfit(x, y, 1), atol=1e-12, rtol=1e-12)
    return dict(margins=margins, means=means, sem=sem, take=take,
                realization_means=means_by_realization, provenance=provenance,
                csv_sha256=sha256(source))


def draw(data):
    margins, mean, sem, take = (data[k] for k in ("margins", "means", "sem", "take"))
    fit = data["provenance"]["expected_fit"]
    A, alpha, r2 = fit["A"], fit["alpha"], fit["log_space_R2"]
    lo, hi = data["provenance"]["fit_margin_interval"]
    g_lo, g_hi, gain = 1 - hi, 1 - lo, 1 - margins

    def evaluate(g):
        return A * (1 - np.asarray(g)) ** (-alpha)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .9,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.245, top=.88, wspace=.25)
    fitted_g = np.linspace(g_lo, g_hi, 400)
    for index, ax in enumerate(axes):
        ax.set_ylim(.24, .81)
        ax.set_xlabel("Gain, $G$")
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(direction="out", length=3.5, width=.8)
        ax.text(-.15, 1.065, "AB"[index], transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")
        if index:
            ax.axvspan(g_lo, g_hi, color="#dce2e8", alpha=.50, zorder=0)
            ax.text((g_lo + g_hi) / 2, .765, "Fit\ninterval", ha="center",
                    va="top", fontsize=8.5, color="#555b62")
            ax.errorbar(gain[~take], mean[~take], yerr=sem[~take], fmt="o", ms=4.4,
                color=GRAY, ecolor=GRAY, capsize=2, elinewidth=.9, capthick=.9,
                mec="white", mew=.45, zorder=3)
            for bounds in [(gain.min(), g_lo), (g_hi, gain.max())]:
                gg = np.linspace(*bounds, 300)
                ax.plot(gg, evaluate(gg), ":", color="#333333", lw=1.6, zorder=2)
            ax.set(xlim=(.692, 1.001), xticks=[.70, .80, .90, .95, .99],
                   title="Full simulated gain range")
        else:
            ax.set(xlim=(.927, .997), xticks=[.93, .95, .97, .99], title="Fitted gain range")
            equation = rf"$\mathrm{{fALFF}} = {A:.4f}\,(1-G)^{{-{alpha:.4f}}}$"
            ax.text(.045, .955, equation + "\n" + rf"$R^2_{{\mathrm{{log}}}} = {r2:.3f}$",
                    transform=ax.transAxes, va="top", fontsize=10.5, linespacing=1.7)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.plot(fitted_g, evaluate(fitted_g), color="#222222", lw=1.6, zorder=2)
        ax.errorbar(gain[take], mean[take], yerr=sem[take], fmt="o", ms=4.6,
            color=ORANGE, ecolor=ORANGE, capsize=2, elinewidth=.9, capthick=.9,
            mec="white", mew=.45, zorder=4)
    axes[0].set_ylabel("Simulated fALFF")
    handles = [Line2D([], [], color=ORANGE, marker="o", ls="none", ms=4.6),
        Line2D([], [], color=GRAY, marker="o", ls="none", ms=4.6),
        Line2D([], [], color="#222222", lw=1.6),
        Line2D([], [], color="#333333", lw=1.6, ls=":")]
    fig.legend(handles, ["Fitted-range means", "Outside fit range", "Power-law fit", "Extrapolation"],
        loc="lower center", bbox_to_anchor=(.53, .025), ncol=4, frameon=False,
        fontsize=8.5, handlelength=1.7, columnspacing=1.3, handletextpad=.5)
    return fig


def make_caption(data):
    fit = data["provenance"]["expected_fit"]
    A, alpha, r2 = fit["A"], fit["alpha"], fit["log_space_R2"]
    lo, hi = data["provenance"]["fit_margin_interval"]
    gain = 1 - data["margins"]
    return f"""**Figure S3. Nonlinear dependence of simulated fALFF on recurrent gain at τ = 120 ms.**

Simulations used random recurrent networks with N = 400 units, connection probability p = 0.2 and neuronal time constant τ = 120 ms. Activity was observed through the 1.3-s effective filter used in the revised Figure 4. Points show mean fALFF across 14 independent network-and-input realizations, after averaging four 300-sample observation windows within each realization; error bars indicate SEM across the 14 realization means.

**(A)** Mean fALFF over the fitted gain interval, G = {1-hi:.3f}–{1-lo:.3f} (actual distance to instability G′ = 1 − G = 0.006–0.07). The solid curve shows the existing power-law approximation, $\\mathrm{{fALFF}} \\approx {A:.4f}(1-G)^{{-{alpha:.4f}}}$, obtained by ordinary least-squares regression of log₁₀(mean fALFF) on log₁₀(G′), with equal weight for the eight simulated gain levels in this interval (log-space R² = {r2:.3f}).

**(B)** The same ensemble over the full simulated range, G = {gain.min():.3f}–{gain.max():.3f} (G′ = 0.003–0.3). Shading and orange points identify the interval fitted in A; gray points lie outside that interval. The solid segment repeats the same fitted curve, and dotted segments show its extrapolation, without refitting. The broader simulation curve departs from this finite-range approximation, showing flattening toward both ends of the sampled range. The approximation is descriptive; it does not establish universal critical scaling or replace the numerical calibration used for the cortical map. This figure presents the stationary ensemble mean rather than the three-realization sampling display in main Figure 4C.
"""


def generate(output_dir, data_dir=DATA, formats=("png", "pdf", "svg")):
    formats = tuple(formats)
    if not formats or any(ext not in ("png", "pdf", "svg") for ext in formats):
        raise ValueError("Choose one or more formats from png,pdf,svg")
    data = load_data(data_dir)
    output_dir = Path(output_dir).resolve()
    if output_dir == Path(data_dir).resolve():
        raise ValueError("Output directory must differ from the source-data directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw(data)
    try:
        for ext in formats:
            fig.savefig(output_dir / f"figure_S3.{ext}", dpi=400)
    finally:
        plt.close(fig)
    (output_dir / "caption.md").write_text(make_caption(data), encoding="utf-8")
    fit = data["provenance"]["expected_fit"]
    with (output_dir / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Gprime_actual", "G_actual", "mean_fALFF", "SEM_across_14_networks",
                         "within_fit_range", "powerlaw_prediction_same_fit"])
        for m, y, se, take in zip(data["margins"], data["means"], data["sem"], data["take"]):
            writer.writerow([m, 1-m, y, se, bool(take), fit["A"] * m**(-fit["alpha"])])
    verification = {"tau_ms": 120, "realizations": 14, "windows_per_realization": 4,
        "fit_gain_points": 8, "full_gain_points": 15, "fit": fit,
        "csv_sha256": data["csv_sha256"], "source_checksum_verified": True,
        "archived_fit_and_independent_polyfit_verified": True,
        "SEM_unit": "network-and-input realization after within-realization window averaging",
        "one_fit_shared_by_both_panels": True, "no_empirical_data_or_new_simulations": True,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "files_sha256": {name: sha256(output_dir / name) for name in
            [*(f"figure_S3.{ext}" for ext in formats), "caption.md", "source_data.csv"]}}
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Directory containing the verified simulation CSV and provenance.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "figure_S3")
    parser.add_argument("--formats", default="png,pdf,svg", help="Comma-separated subset of png,pdf,svg")
    args = parser.parse_args(argv)
    out = generate(args.output_dir, args.data_dir, args.formats.split(","))
    print(f"Figure S3, caption and verification written to {out}")


if __name__ == "__main__":
    main()
