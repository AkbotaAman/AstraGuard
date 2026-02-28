import os
import sys
import streamlit as st
import matplotlib.pyplot as plt

# Make src importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from telemetry_simulator import simulate_telemetry, inject_failure
from ai_detector import AnomalyDetector
from comparison import compare_ai_vs_human

st.set_page_config(page_title="AstraGuard Demo", layout="wide")

st.title("AstraGuard — Interactive MVP Demo")
st.caption("AI-powered autonomous onboard protection system (simulation-based MVP).")

# Sidebar controls
st.sidebar.header("Simulation Controls")

failure_type = st.sidebar.selectbox(
    "Failure Type",
    ["thermal_runaway", "power_drain", "comm_drop"],
    index=0,
)

failure_start = st.sidebar.slider("Failure Start (t)", min_value=20, max_value=280, value=160, step=5)
severity = st.sidebar.slider("Severity", min_value=0.5, max_value=2.0, value=1.2, step=0.1)
human_delay = st.sidebar.slider("Human Response Delay (steps)", min_value=0, max_value=40, value=12, step=1)

contamination = st.sidebar.slider("Anomaly Detector Sensitivity", min_value=0.01, max_value=0.10, value=0.04, step=0.01)

run = st.sidebar.button("Run Simulation")

if run:
    # 1) Training telemetry (normal)
    train_df = simulate_telemetry(n_steps=450, seed=1, noise_scale=1.0)

    # 2) Mission telemetry + failure
    mission_df = simulate_telemetry(n_steps=300, seed=7, noise_scale=1.1)
    failed_df = inject_failure(mission_df, failure_type=failure_type, start=failure_start, severity=severity)

    # 3) Fit detector
    detector = AnomalyDetector(contamination=contamination, random_state=42)
    detector.fit(train_df)

    # 4) Compare AI vs human delay
    results = compare_ai_vs_human(failed_df, detector, human_delay_steps=human_delay)
    ai_res = results["ai"]
    human_res = results["human"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Summary")
        st.write(f"**AI trigger step:** {ai_res['trigger_step']}")
        st.write(f"**Human trigger step:** {human_res['trigger_step']}")
        st.write(f"**AI total damage:** {ai_res['total_damage']:.2f}")
        st.write(f"**Human total damage:** {human_res['total_damage']:.2f}")
        st.write(f"**AI survival score:** {ai_res['survival_score']:.2f}")
        st.write(f"**Human survival score:** {human_res['survival_score']:.2f}")

    with col2:
        delta = human_res["total_damage"] - ai_res["total_damage"]
        st.subheader("Impact")
        st.metric("Damage Reduced (Human − AI)", f"{delta:.2f}")
        if human_res["total_damage"] > 0:
            pct = 100.0 * delta / human_res["total_damage"]
            st.metric("Relative Reduction", f"{pct:.1f}%")

    st.divider()

    # ---- Plot 1: Telemetry
    st.subheader("Telemetry (with injected failure)")
    fig1 = plt.figure()
    t = failed_df["t"].to_numpy()
    plt.plot(t, failed_df["battery"], label="battery")
    plt.plot(t, failed_df["temperature"], label="temperature")
    plt.plot(t, failed_df["signal"], label="signal")
    plt.plot(t, failed_df["cpu_load"], label="cpu_load")
    plt.xlabel("t")
    plt.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # ---- Plot 2: Anomaly scores + triggers
    st.subheader("Anomaly Scores & Trigger Time")
    fig2 = plt.figure()
    plt.plot(t, ai_res["anomaly_scores"], label="anomaly_score")
    if ai_res["trigger_step"] is not None:
        plt.axvline(ai_res["trigger_step"], linestyle="--", label="AI trigger")
    if human_res["trigger_step"] is not None:
        plt.axvline(human_res["trigger_step"], linestyle="--", label="Human trigger")
    plt.xlabel("t")
    plt.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # ---- Plot 3: Cumulative damage (using existing traces)
    st.subheader("AI vs Human — Cumulative Mission Damage (proxy)")
    ai_trace = ai_res["trace"]
    human_trace = human_res["trace"]

    def quick_damage(df):
        import numpy as np
        dmg = (
            np.maximum(0, 40 - df["battery"].to_numpy()) * 0.6
            + np.maximum(0, df["temperature"].to_numpy() - 45) * 1.2
            + np.maximum(0, 50 - df["signal"].to_numpy()) * 0.7
            + np.maximum(0, df["cpu_load"].to_numpy() - 80) * 0.4
        )
        return dmg

    fig3 = plt.figure()
    plt.plot(t, quick_damage(human_trace).cumsum(), label="Human delayed response")
    plt.plot(t, quick_damage(ai_trace).cumsum(), label="AI instant response")
    plt.xlabel("t")
    plt.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()
    st.success("Demo run completed. Adjust parameters and run again to test different scenarios.")
else:
    st.info("Set parameters on the left and click **Run Simulation**.")
