import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="AstraGuard — Interactive MVP Demo", layout="wide")

st.title("AstraGuard — Interactive MVP Demo")
st.caption("Simulation-based MVP: compare Human vs AI anomaly detection & recovery (toy model).")

# --- Controls ---
with st.sidebar:
    st.header("Simulation Controls")

    failure_start = st.slider("Failure start (t)", 0, 300, 160)
    severity = st.slider("Severity", 0.1, 2.0, 1.2, 0.1)
    anomaly_rate = st.slider("Anomaly frequency (events / 1000 steps)", 0, 40, 12)
    response_delay = st.slider("Human response delay (steps)", 0, 50, 12)

    st.divider()
    st.subheader("AI Controls")
    ai_enabled = st.toggle("Enable AI mode", value=True)
    ai_sensitivity = st.slider("AI detection sensitivity", 0.0, 1.0, 0.65, 0.05)
    ai_response_speed = st.slider("AI response speed", 0.0, 1.0, 0.60, 0.05)
    auto_reconfig = st.toggle("Auto reconfigure subsystem", value=True)

    st.divider()
    seed = st.number_input("Random seed", value=42, step=1)
    runs = st.slider("Monte-Carlo runs", 5, 50, 20)

# --- Core simulation (toy but responsive & consistent) ---
rng = np.random.default_rng(int(seed))

def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))

def simulate_one(is_ai: bool):
    # baseline human detection/recovery depend on severity + anomaly rate
    base_mttd = 25 + 1.4 * anomaly_rate + 18 * severity
    base_mttr = 60 + 2.0 * anomaly_rate + 35 * severity + response_delay

    # AI improves detection & recovery
    if is_ai and ai_enabled:
        det_gain = 0.15 + 0.70 * ai_sensitivity          # up to ~0.85
        rec_gain = 0.10 + 0.60 * ai_response_speed        # up to ~0.70
        if auto_reconfig:
            rec_gain += 0.10  # extra recovery gain from auto reconfig

        mttd = base_mttd * (1.0 - det_gain)
        mttr = base_mttr * (1.0 - rec_gain)
    else:
        mttd = base_mttd
        mttr = base_mttr

    # add noise so sliders + runs look real
    mttd *= rng.normal(1.0, 0.08)
    mttr *= rng.normal(1.0, 0.10)

    # risk grows with severity, anomaly_rate and slow reaction
    risk = (severity * 220) + (anomaly_rate * 12) + (mttd * 4.0) + (mttr * 2.5)

    # mission survival score (higher is better)
    survival = 1000.0 - risk
    survival = clip(survival, 0.0, 1000.0)

    return mttd, mttr, risk, survival

# Monte Carlo
rows = []
for _ in range(int(runs)):
    hmttd, hmttr, hrisk, hsurv = simulate_one(False)
    amttd, amttr, arisk, asurv = simulate_one(True)
    rows.append([hmttd, hmttr, hrisk, hsurv, amttd, amttr, arisk, asurv])

df = pd.DataFrame(rows, columns=[
    "Human MTTD", "Human MTTR", "Human Risk", "Human Survival",
    "AI MTTD", "AI MTTR", "AI Risk", "AI Survival"
])

# Aggregate
human = df[["Human MTTD","Human MTTR","Human Risk","Human Survival"]].mean()
ai = df[["AI MTTD","AI MTTR","AI Risk","AI Survival"]].mean()

impact_damage_reduced = human["Human Risk"] - ai["AI Risk"]
survival_gain = ai["AI Survival"] - human["Human Survival"]

# --- UI output ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Survival Score", f"{ai['AI Survival']:.1f}", f"{survival_gain:+.1f} vs Human")

with col2:
    st.metric("Human Survival Score", f"{human['Human Survival']:.1f}")

with col3:
    st.metric("Impact / Risk Reduced (Human − AI)", f"{impact_damage_reduced:.1f}")

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("MTTD / MTTR (avg over runs)")
    st.write(f"**Human MTTD:** {human['Human MTTD']:.1f} steps")
    st.write(f"**AI MTTD:** {ai['AI MTTD']:.1f} steps")
    st.write(f"**Human MTTR:** {human['Human MTTR']:.1f} steps")
    st.write(f"**AI MTTR:** {ai['AI MTTR']:.1f} steps")

with c2:
    st.subheader("Risk breakdown (toy model)")
    st.caption("Lower risk is better. This MVP demonstrates relative improvement.")
    st.dataframe(df.head(10), use_container_width=True)

st.info("Note: This is a simulation-based MVP. The goal is to demonstrate AI-driven reduction in detection and recovery time under anomalies.")
