from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_telemetry(
    n_steps: int = 300,
    seed: int = 42,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Generate synthetic spacecraft telemetry time series.

    Columns:
      - t (step index)
      - battery (0..100)
      - temperature (C)
      - signal (0..100)
      - cpu_load (0..100)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # Base trends + periodic components
    battery = 100 - 0.02 * t + 0.6 * np.sin(t / 30.0)
    temperature = 35 + 0.8 * np.sin(t / 18.0) + 0.5 * np.cos(t / 50.0)
    signal = 85 + 3.0 * np.sin(t / 40.0) - 0.02 * (t / 10.0)
    cpu = 35 + 8.0 * np.sin(t / 10.0) + 5.0 * np.cos(t / 17.0)

    # Add noise
    battery += rng.normal(0, 0.25 * noise_scale, size=n_steps)
    temperature += rng.normal(0, 0.35 * noise_scale, size=n_steps)
    signal += rng.normal(0, 0.9 * noise_scale, size=n_steps)
    cpu += rng.normal(0, 1.2 * noise_scale, size=n_steps)

    df = pd.DataFrame(
        {
            "t": t,
            "battery": battery,
            "temperature": temperature,
            "signal": signal,
            "cpu_load": cpu,
        }
    )

    # Clip to realistic bounds
    df["battery"] = df["battery"].clip(0, 100)
    df["signal"] = df["signal"].clip(0, 100)
    df["cpu_load"] = df["cpu_load"].clip(0, 100)

    return df


def inject_failure(
    df: pd.DataFrame,
    failure_type: str = "thermal_runaway",
    start: int = 160,
    severity: float = 1.0,
) -> pd.DataFrame:
    """
    Inject a failure pattern into telemetry.

    failure_type:
      - "thermal_runaway": temperature rises, cpu rises a bit
      - "power_drain": battery drops faster, cpu may dip
      - "comm_drop": signal decreases sharply, cpu rises
    """
    out = df.copy()
    n = len(out)
    start = max(0, min(start, n - 1))

    idx = np.arange(start, n)
    k = idx - start

    if failure_type == "thermal_runaway":
        out.loc[idx, "temperature"] += severity * (0.05 * k + 0.8 * np.sin(k / 7.0))
        out.loc[idx, "cpu_load"] += severity * (0.03 * k)
    elif failure_type == "power_drain":
        out.loc[idx, "battery"] -= severity * (0.06 * k + 0.4 * np.abs(np.sin(k / 10.0)))
        out.loc[idx, "cpu_load"] -= severity * (0.01 * k)
    elif failure_type == "comm_drop":
        out.loc[idx, "signal"] -= severity * (0.18 * k + 3.0 * np.abs(np.sin(k / 6.0)))
        out.loc[idx, "cpu_load"] += severity * (0.02 * k)
    else:
        raise ValueError(f"Unknown failure_type: {failure_type}")

    # Re-clip
    out["battery"] = out["battery"].clip(0, 100)
    out["signal"] = out["signal"].clip(0, 100)
    out["cpu_load"] = out["cpu_load"].clip(0, 100)

    return out
