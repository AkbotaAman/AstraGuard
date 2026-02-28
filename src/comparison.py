from __future__ import annotations
import numpy as np
import pandas as pd

from .ai_detector import AnomalyDetector
from .controller import diagnose_failure, apply_recovery, mission_damage


def run_scenario(
    df: pd.DataFrame,
    detector: AnomalyDetector,
    human_delay_steps: int = 12,
    ai_enabled: bool = True,
    anomaly_threshold: float | None = None,
) -> dict:
    """
    Simulates an online control loop.
    - If ai_enabled: recovery triggered immediately on anomaly.
    - If not: recovery triggered after human_delay_steps after first anomaly.
    Returns metrics and the resulting telemetry trace.
    """
    trace = df.copy()
    n = len(trace)

    anomaly_scores = detector.score(trace)
    if anomaly_threshold is None:
        anomaly_threshold = np.quantile(anomaly_scores, 0.96)
    is_anom = anomaly_scores >= anomaly_threshold

    first_anom = int(np.argmax(is_anom)) if is_anom.any() else None
    trigger_step = None
    if first_anom is not None:
        trigger_step = first_anom if ai_enabled else min(n - 1, first_anom + human_delay_steps)

    # Apply "recovery" effects from trigger_step onward (simulated)
    recovered = []
    damage_series = []
    action_taken = ["none"] * n

    for i in range(n):
        row = trace.iloc[i].copy()

        if trigger_step is not None and i >= trigger_step:
            # diagnose current state, apply recovery
            failure = diagnose_failure(row)
            row = apply_recovery(row, failure)
            action_taken[i] = failure
            recovered.append(True)
        else:
            recovered.append(False)

        # write back row (simulate mitigation impact)
        trace.iloc[i] = row
        damage_series.append(mission_damage(row))

    total_damage = float(np.sum(damage_series))
    survival_score = float(max(0.0, 1000.0 - total_damage))  # simple score
    return {
        "trace": trace,
        "anomaly_scores": anomaly_scores,
        "is_anomaly": is_anom,
        "trigger_step": trigger_step,
        "total_damage": total_damage,
        "survival_score": survival_score,
        "action_taken": action_taken,
    }


def compare_ai_vs_human(
    df_with_failure: pd.DataFrame,
    detector: AnomalyDetector,
    human_delay_steps: int = 12,
) -> dict:
    ai = run_scenario(df_with_failure, detector, human_delay_steps=human_delay_steps, ai_enabled=True)
    human = run_scenario(df_with_failure, detector, human_delay_steps=human_delay_steps, ai_enabled=False)

    return {"ai": ai, "human": human}
