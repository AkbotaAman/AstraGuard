from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from telemetry_simulator import simulate_telemetry, inject_failure
from ai_detector import AnomalyDetector
from comparison import compare_ai_vs_human


def ensure_results_dir() -> str:
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_results(base_df, failed_df, ai_res, human_res, out_dir: str) -> None:
    t = failed_df["t"].to_numpy()

    # 1) Telemetry plot
    plt.figure()
    plt.plot(t, failed_df["battery"], label="battery")
    plt.plot(t, failed_df["temperature"], label="temperature")
    plt.plot(t, failed_df["signal"], label="signal")
    plt.plot(t, failed_df["cpu_load"], label="cpu_load")
    plt.title("Telemetry with Injected Failure")
    plt.xlabel("t")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "telemetry.png"), dpi=160)
    plt.close()

    # 2) Anomaly score plot (from AI scenario)
    plt.figure()
    plt.plot(t, ai_res["anomaly_scores"], label="anomaly_score")
    if ai_res["trigger_step"] is not None:
        plt.axvline(ai_res["trigger_step"], linestyle="--", label="AI trigger")
    if human_res["trigger_step"] is not None:
        plt.axvline(human_res["trigger_step"], linestyle="--", label="Human trigger")
    plt.title("Anomaly Scores and Trigger Time")
    plt.xlabel("t")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "anomaly_scores.png"), dpi=160)
    plt.close()

    # 3) Damage comparison
    # Recompute per-step damage proxy from anomaly scores for visualization simplicity:
    # We'll approximate damage as normalized anomaly score + penalize battery drop etc.
    # (The real metric used in code is mission_damage; this plot is illustrative.)
    ai_trace = ai_res["trace"]
    human_trace = human_res["trace"]

    def quick_damage(df):
        dmg = (
            np.maximum(0, 40 - df["battery"].to_numpy()) * 0.6
            + np.maximum(0, df["temperature"].to_numpy() - 45) * 1.2
            + np.maximum(0, 50 - df["signal"].to_numpy()) * 0.7
            + np.maximum(0, df["cpu_load"].to_numpy() - 80) * 0.4
        )
        return dmg

    plt.figure()
    plt.plot(t, np.cumsum(quick_damage(human_trace)), label="Human delayed response (cumulative damage)")
    plt.plot(t, np.cumsum(quick_damage(ai_trace)), label="AI instant response (cumulative damage)")
    plt.title("AI vs Human: Cumulative Mission Damage")
    plt.xlabel("t")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ai_vs_human_damage.png"), dpi=160)
    plt.close()


def main():
    out_dir = ensure_results_dir()

    # 1) Generate normal telemetry for training
    train_df = simulate_telemetry(n_steps=400, seed=1, noise_scale=1.0)

    # 2) Generate mission telemetry and inject a failure
    mission_df = simulate_telemetry(n_steps=300, seed=7, noise_scale=1.1)
    failed_df = inject_failure(mission_df, failure_type="thermal_runaway", start=160, severity=1.2)

    # 3) Fit anomaly detector on normal telemetry
    detector = AnomalyDetector(contamination=0.04, random_state=42)
    detector.fit(train_df)

    # 4) Compare AI vs Human delay
    results = compare_ai_vs_human(failed_df, detector, human_delay_steps=12)
    ai_res = results["ai"]
    human_res = results["human"]

    # 5) Print summary
    print("=== AstraGuard MVP Summary ===")
    print(f"AI trigger step: {ai_res['trigger_step']}")
    print(f"Human trigger step: {human_res['trigger_step']}")
    print(f"AI total damage: {ai_res['total_damage']:.2f} | survival score: {ai_res['survival_score']:.2f}")
    print(f"Human total damage: {human_res['total_damage']:.2f} | survival score: {human_res['survival_score']:.2f}")
    print(f"Saved plots to: {out_dir}/")

    # 6) Save plots
    plot_results(train_df, failed_df, ai_res, human_res, out_dir)


if __name__ == "__main__":
    main()
