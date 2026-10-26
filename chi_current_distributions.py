#!/usr/bin/env python3
"""Plot time-averaged homodyne-current distributions for several chi values."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qfi-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


DEFAULT_CHI_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)


def simulate_current_averages(
    chi: float,
    times: np.ndarray,
    *,
    trajectories: int,
    internal_dt: float,
    seed: int,
    cpus: int,
    transient_time: float,
) -> np.ndarray:
    """Return one post-transient time-averaged current per trajectory."""

    cutoff = 5
    cavity_frequency = 1.0
    qubit_frequency = 1.0
    drive = 0.2 + 0.0j
    homodyne_phase = np.pi / 4.0
    efficiency = 0.8
    kappa = 1.0
    gamma = 0.01
    gamma_phi = 0.01

    identity_cavity = qt.qeye(cutoff)
    identity_qubit = qt.qeye(2)
    sigma_z = qt.tensor(identity_cavity, qt.sigmaz())
    sigma_minus = qt.tensor(identity_cavity, qt.sigmam())
    cavity = qt.tensor(qt.destroy(cutoff), identity_qubit)
    number = cavity.dag() * cavity

    hamiltonian = (
        cavity_frequency * number
        + 0.5 * qubit_frequency * sigma_z
        + chi * number * sigma_z
        + drive * cavity.dag()
        + drive.conjugate() * cavity
    )

    initial_state = qt.tensor(
        qt.basis(cutoff, 0),
        (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
    )
    unmonitored = [
        np.sqrt((1.0 - efficiency) * kappa) * cavity,
        np.sqrt(gamma) * sigma_minus,
        np.sqrt(gamma_phi / 2.0) * sigma_z,
    ]
    monitored = [
        np.sqrt(efficiency * kappa)
        * np.exp(-1j * homodyne_phase)
        * cavity
    ]

    result = qt.smesolve(
        hamiltonian,
        initial_state,
        times,
        c_ops=unmonitored,
        sc_ops=monitored,
        ntraj=trajectories,
        seeds=seed,
        options={
            "method": "rouchon",
            "dt": internal_dt,
            "store_states": False,
            "store_measurement": True,
            "keep_runs_results": True,
            "progress_bar": "",
            "map": "parallel" if cpus > 1 else "serial",
            "num_cpus": min(cpus, trajectories),
        },
    )

    # QuTiP stores the homodyne current over each output interval.  Thus the
    # last axis has len(times) - 1 entries and entry j belongs to
    # [times[j], times[j + 1]].
    measurements = np.asarray(result.measurement)
    current_runs = measurements[:, 0, :].real
    first_interval = int(np.searchsorted(times[:-1], transient_time, side="left"))
    if first_interval >= current_runs.shape[1]:
        raise ValueError("transient_time must be smaller than final_time")
    return current_runs[:, first_interval:].mean(axis=1)


def save_plot(
    path: Path,
    data_by_chi: dict[float, np.ndarray],
    *,
    final_time: float,
    output_dt: float,
    internal_dt: float,
    transient_time: float,
    trajectories: int,
) -> None:
    """Use stacked densities and connected means to expose the shift."""

    chi_values = np.asarray(list(data_by_chi), dtype=float)
    samples = list(data_by_chi.values())
    means = np.asarray([values.mean() for values in samples])
    sample_stds = np.asarray([values.std(ddof=1) for values in samples])
    confidence_95 = 1.96 * sample_stds / np.sqrt(
        np.asarray([len(values) for values in samples])
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(data_by_chi)))
    baselines = np.arange(len(samples), dtype=float)
    all_values = np.concatenate(samples)
    value_range = float(np.ptp(all_values))
    padding = 0.08 * value_range
    current_grid = np.linspace(
        float(all_values.min()) - padding,
        float(all_values.max()) + padding,
        500,
    )

    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)

    for baseline, color, values in zip(baselines, colors, samples):
        bandwidth = max(
            1.06 * float(values.std(ddof=1)) * len(values) ** (-0.2),
            np.finfo(float).eps,
        )
        scaled_distance = (current_grid[:, None] - values[None, :]) / bandwidth
        density = np.exp(-0.5 * scaled_distance**2).mean(axis=1)
        density /= bandwidth * np.sqrt(2.0 * np.pi)
        ridge_height = 0.72 * density / density.max()
        axis.fill_between(
            current_grid,
            baseline,
            baseline + ridge_height,
            color=color,
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            current_grid,
            baseline + ridge_height,
            color=color,
            linewidth=2.0,
            zorder=2,
        )
        axis.plot(
            [current_grid[0], current_grid[-1]],
            [baseline, baseline],
            color="0.82",
            linewidth=0.8,
            zorder=0,
        )

    axis.axvline(
        means[0],
        color=colors[0],
        linestyle="--",
        linewidth=1.3,
        alpha=0.6,
        zorder=0,
    )
    axis.plot(
        means,
        baselines,
        color="0.18",
        linewidth=2.2,
        zorder=4,
        label="Mean trajectory",
    )
    for baseline, color, mean, interval in zip(
        baselines, colors, means, confidence_95
    ):
        axis.errorbar(
            mean,
            baseline,
            xerr=interval,
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=8,
            capsize=4,
            linewidth=2,
            zorder=5,
        )

    total_shift = means[-1] - means[0]
    bracket_height = -0.32
    axis.annotate(
        "",
        xy=(means[-1], bracket_height),
        xytext=(means[0], bracket_height),
        arrowprops={"arrowstyle": "<->", "color": "0.25", "linewidth": 1.5},
    )
    axis.text(
        0.5 * (means[0] + means[-1]),
        bracket_height - 0.10,
        rf"Total mean shift: {total_shift:.4f}",
        ha="center",
        va="top",
        fontsize=10,
    )

    axis.set_yticks(baselines, [rf"$\chi={chi:.2f}$" for chi in chi_values])
    axis.set_ylim(-0.58, baselines[-1] + 0.9)
    axis.set_xlabel("Post-transient time-averaged homodyne current")
    axis.set_ylabel("Distribution")
    axis.set_title(r"Homodyne-current distribution shifts left as $\chi$ increases")
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.99,
        0.98,
        (
            rf"Integrator step: $dt={internal_dt:g}$" "\n"
            rf"Record interval: $\Delta t={output_dt:g}$" "\n"
            rf"Trajectories: $N_{{\rm traj}}={trajectories}$ per $\chi$" "\n"
            rf"Averaging window: ${transient_time:g}\leq t\leq{final_time:g}$"
            "\n\n"
            "Filled curves: density\nPoints/bars: mean ± 95% CI"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_summary(path: Path, data_by_chi: dict[float, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("chi", "trajectories", "mean", "variance", "std_ddof_1"))
        for chi, values in data_by_chi.items():
            writer.writerow(
                (
                    chi,
                    len(values),
                    values.mean(),
                    values.var(ddof=1),
                    values.std(ddof=1),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chi-values", type=float, nargs="+", default=DEFAULT_CHI_VALUES)
    parser.add_argument("--final-time", type=float, default=1000.0)
    parser.add_argument("--output-dt", type=float, default=1.0)
    parser.add_argument("--internal-dt", type=float, default=0.001)
    parser.add_argument("--transient-time", type=float, default=50.0)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpus", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("chi_current_ridgeline.png"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("chi_current_distribution_summary.csv"),
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=Path("chi_current_time_averages.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_time <= 0 or args.output_dt <= 0 or args.internal_dt <= 0:
        raise ValueError("time values and time steps must be positive")
    if args.trajectories < 2 or args.cpus < 1:
        raise ValueError("use at least 2 trajectories and 1 CPU")
    interval_count = round(args.final_time / args.output_dt)
    if not np.isclose(interval_count * args.output_dt, args.final_time):
        raise ValueError("final_time must be an integer multiple of output_dt")
    times = np.linspace(0.0, args.final_time, interval_count + 1)

    data_by_chi: dict[float, np.ndarray] = {}
    start = time.perf_counter()
    for chi in args.chi_values:
        chi_start = time.perf_counter()
        print(f"Running chi={chi:.2f} ...", flush=True)
        data_by_chi[chi] = simulate_current_averages(
            chi,
            times,
            trajectories=args.trajectories,
            internal_dt=args.internal_dt,
            seed=args.seed,
            cpus=args.cpus,
            transient_time=args.transient_time,
        )
        values = data_by_chi[chi]
        print(
            f"  mean={values.mean():.6f}, SD={values.std(ddof=1):.6f}, "
            f"elapsed={time.perf_counter() - chi_start:.1f}s",
            flush=True,
        )

    save_plot(
        args.figure_output,
        data_by_chi,
        final_time=args.final_time,
        output_dt=args.output_dt,
        internal_dt=args.internal_dt,
        transient_time=args.transient_time,
        trajectories=args.trajectories,
    )
    save_summary(args.summary_output, data_by_chi)
    np.savez(
        args.samples_output,
        **{f"chi_{chi:.2f}": values for chi, values in data_by_chi.items()},
    )
    print(f"Total elapsed: {time.perf_counter() - start:.1f}s")
    print(f"Figure:  {args.figure_output.resolve()}")
    print(f"Summary: {args.summary_output.resolve()}")
    print(f"Samples: {args.samples_output.resolve()}")


if __name__ == "__main__":
    main()
