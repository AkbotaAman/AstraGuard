import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AstraGuard – Interactive MVP", layout="wide")

st.title("AstraGuard – Interactive MVP Demo")
st.caption("AI-powered autonomous onboard protection system (simulation-based MVP).")

# --------- Controls ----------
with st.sidebar:
    st.header("Simulation Controls")

    failure_start = st.slider("Failure start (t)", 0, 280, 160, 1)
    severity = st.slider("Severity", 0.50, 2.00, 1.20, 0.01)
    human_delay = st.slider("Human response delay (steps)", 0, 40, 12, 1)
    ai_sensitivity = st.slider("AI anomaly detection sensitivity", 0.10, 1.00, 0.60, 0.01)

    run = st.button("Run Simulation")

# --------- Simulation core ----------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def simulate(failure_start, severity, human_delay, ai_sensitivity, T=300, seed=42):
    rng = np.random.default_rng(seed)

    t = np.arange(T)

    # baseline telemetry + noise
    telemetry = 0.02 * rng.normal(size=T)

    # anomaly pattern: spike + drift after failure_start
    telemetry[failure_start:] += (0.15 * severity) + (0.002 * severity) * (t[failure_start:] - failure_start)
    telemetry[failure_start:failure_start+5] += 0.35 * severity  # short spike

    # detection probability grows with severity and sensitivity
    # (not ML training, but a proxy that mimics "stronger detector => earlier/likelier detection")
    detect_p = sigmoid((severity * 2.2 + ai_sensitivity * 3.0) - 4.0)
    detected = rng.random() < detect_p

    # response times
    # AI reacts faster when sensitivity is higher (bounded)
    ai_delay = int(max(1, round(human_delay * (0.55 - 0.35 * ai_sensitivity))))

    # damage model: bigger severity + slower response => more damage
    # if not detected => huge delay penalty
    def damage(delay, detected_flag):
        effective_delay = delay if detected_flag else delay + 35
        base = 160 * (severity ** 1.35)
        return base * (1.0 + effective_delay / 18.0)

    dmg_human = damage(human_delay, True)           # human always "notices", but slow
    dmg_ai = damage(ai_delay, detected)             # AI may miss at low sensitivity

    # survival score (0..1000)
    human_survival = max(0.0, 1000.0 - dmg_human)
    ai_survival = max(0.0, 1000.0 - dmg_ai)

    impact = ai_survival - human_survival  # positive = AI better

    # mark response times on timeline
    human_action_t = min(T - 1, failure_start + human_delay)
    ai_action_t = min(T - 1, failure_start + ai_delay) if detected else None

    return {
        "t": t,
        "telemetry": telemetry,
        "human_survival": human_survival,
        "ai_survival": ai_survival,
        "impact": impact,
        "detected": detected,
        "human_action_t": human_action_t,
        "ai_action_t": ai_action_t
    }

# --------- Run / display ----------
if run:
    out = simulate(failure_start, severity, human_delay, ai_sensitivity)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI survival score", f"{out['ai_survival']:.2f}")
    c2.metric("Human survival score", f"{out['human_survival']:.2f}")
    c3.metric("Impact (AI - Human)", f"{out['impact']:.2f}")
    c4.metric("AI detected anomaly?", "YES" if out["detected"] else "NO")

    st.divider()

    fig = plt.figure()
    plt.plot(out["t"], out["telemetry"])
    plt.axvline(failure_start, linestyle="--")
    plt.axvline(out["human_action_t"], linestyle="--")
    if out["ai_action_t"] is not None:
        plt.axvline(out["ai_action_t"], linestyle="--")
    plt.title("Telemetry timeline (anomaly + response)")
    plt.xlabel("t (steps)")
    plt.ylabel("telemetry")
    st.pyplot(fig)

    st.caption(
        "Interpretation: higher severity and higher human delay reduce survival. "
        "Higher AI sensitivity usually improves outcome, but may still miss at low sensitivity."
    )
else:
    st.info("Adjust parameters on the left and click **Run Simulation**.")
