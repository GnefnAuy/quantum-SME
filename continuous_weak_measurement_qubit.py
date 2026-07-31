#!/usr/bin/env python3
"""Simulate a continuously monitored qubit with a Rouchon update."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qfi-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def expect(operator: np.ndarray, state: np.ndarray) -> float:
    """Return the real expectation value of an operator."""
    return float(np.real(np.trace(operator @ state)))


def simulate_trajectory(
    total_time: float = 8.0,
    time_step: float = 1e-3,
    measurement_rate: float = 1.0,
    efficiency: float = 0.6,
    rabi_frequency: float = 1.2,
    seed: int | None = 4,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    r"""Simulate one conditional trajectory under homodyne measurement.

    The convention is

        c = sqrt(measurement_rate) sigma_z,
        H = (rabi_frequency / 2) sigma_x,

        d rho = -i[H, rho]dt + D[c]rho dt
                + sqrt(efficiency) H[c]rho dW,

        dY = 2 sqrt(efficiency * measurement_rate)
             <sigma_z> dt + dW.

    The discrete Rouchon-type update preserves positivity. The weak-step
    condition requires ``measurement_rate * time_step << 1``.
    """
    if total_time <= 0.0 or time_step <= 0.0:
        raise ValueError("total_time and time_step must be positive")
    if measurement_rate <= 0.0:
        raise ValueError("measurement_rate must be positive")
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError("efficiency must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    step_count = int(round(total_time / time_step))
    times = np.arange(step_count + 1) * time_step

    if initial_state is None:
        ket = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        state = np.outer(ket, ket.conj())
    else:
        state = np.array(initial_state, dtype=complex)
        if state.shape != (2, 2):
            raise ValueError("initial_state must be a 2x2 density matrix")

    hamiltonian = 0.5 * rabi_frequency * SX
    collapse = np.sqrt(measurement_rate) * SZ
    collapse_dagger_collapse = collapse.conj().T @ collapse

    x = np.empty(step_count + 1)
    y = np.empty(step_count + 1)
    z = np.empty(step_count + 1)
    purity = np.empty(step_count + 1)
    current = np.empty(step_count)
    measurement_increments = np.empty(step_count)

    x[0], y[0], z[0] = (
        expect(SX, state),
        expect(SY, state),
        expect(SZ, state),
    )
    purity[0] = float(np.real(np.trace(state @ state)))

    for index in range(step_count):
        mean_signal = np.sqrt(efficiency) * expect(
            collapse + collapse.conj().T, state
        )
        wiener_increment = np.sqrt(time_step) * rng.normal()
        measurement_increment = mean_signal * time_step + wiener_increment

        update = (
            I2
            - (1j * hamiltonian + 0.5 * collapse_dagger_collapse)
            * time_step
            + np.sqrt(efficiency) * collapse * measurement_increment
        )
        unnormalized_state = update @ state @ update.conj().T
        unnormalized_state += (
            (1.0 - efficiency)
            * (collapse @ state @ collapse.conj().T)
            * time_step
        )
        state = unnormalized_state / np.trace(unnormalized_state)
        state = 0.5 * (state + state.conj().T)

        measurement_increments[index] = measurement_increment
        current[index] = measurement_increment / time_step
        x[index + 1], y[index + 1], z[index + 1] = (
            expect(SX, state),
            expect(SY, state),
            expect(SZ, state),
        )
        purity[index + 1] = float(np.real(np.trace(state @ state)))

    return {
        "time": times,
        "x": x,
        "y": y,
        "z": z,
        "purity": purity,
        "current": current,
        "measurement_increments": measurement_increments,
        "time_step": time_step,
    }


def main() -> None:
    total_time = 8.0
    time_step = 1e-3
    measurement_rate = 1.0
    efficiency = 0.6
    rabi_frequency = 1.2
    trajectory = simulate_trajectory(
        total_time=total_time,
        time_step=time_step,
        measurement_rate=measurement_rate,
        efficiency=efficiency,
        rabi_frequency=rabi_frequency,
    )

    times = np.asarray(trajectory["time"])
    z = np.asarray(trajectory["z"])

    plt.figure(figsize=(9, 5))
    plt.plot(times, trajectory["x"], label=r"$\langle \sigma_x\rangle_c$")
    plt.plot(times, trajectory["y"], label=r"$\langle \sigma_y\rangle_c$")
    plt.plot(times, z, label=r"$\langle \sigma_z\rangle_c$")
    plt.xlabel("Time")
    plt.ylabel("Conditional expectation value")
    plt.title("One trajectory under continuous weak measurement")
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("continuous_weak_measurement_trajectory.png", dpi=180)
    plt.close()

    smoothing_time = 0.08
    window = max(1, int(smoothing_time / time_step))
    smoothed_current = np.convolve(
        trajectory["current"], np.ones(window) / window, mode="same"
    )
    plt.figure(figsize=(9, 4))
    plt.plot(times[:-1], smoothed_current, linewidth=1)
    plt.plot(
        times,
        2.0 * np.sqrt(efficiency * measurement_rate) * z,
        linewidth=2,
        label=r"signal mean $2\sqrt{\eta\Gamma}\langle\sigma_z\rangle_c$",
    )
    plt.xlabel("Time")
    plt.ylabel("Homodyne current")
    plt.title(f"Measurement record, averaged over {smoothing_time:g} time units")
    plt.legend()
    plt.tight_layout()
    plt.savefig("continuous_weak_measurement_record.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(times, trajectory["purity"])
    plt.xlabel("Time")
    plt.ylabel(r"$\mathrm{Tr}(\rho_c^2)$")
    plt.title("Conditional-state purity")
    plt.ylim(0.45, 1.02)
    plt.tight_layout()
    plt.savefig("continuous_weak_measurement_purity.png", dpi=180)
    plt.close()

    print(f"measurement_rate * time_step = {measurement_rate * time_step:.3e}")
    print("Saved continuous weak-measurement trajectory, record, and purity plots")


if __name__ == "__main__":
    main()
