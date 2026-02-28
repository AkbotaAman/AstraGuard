# AstraGuard  
Autonomous AI System for Real-Time Spacecraft Failure Recovery  

---

## Problem

Deep space missions suffer from significant communication delays (up to 20+ minutes to Mars). During emergencies, spacecraft must wait for human instructions from Earth. This delay can lead to critical subsystem failure and mission loss.

Current spacecraft rely heavily on ground control decision-making. In high-risk scenarios, delayed reaction may cause irreversible damage to onboard systems.

---

## Our Solution

AstraGuard is an onboard AI-based autonomous protection system that:

- Detects anomalies in spacecraft telemetry  
- Diagnoses subsystem failures  
- Automatically reconfigures onboard systems  
- Reduces reaction time from minutes to seconds  

The system operates independently from Earth-based control, enabling immediate reaction during critical situations.

---

## AI Core

The system integrates:

- Isolation Forest for anomaly detection  
- Real-time spacecraft telemetry simulation  
- Rule-based autonomous response engine  

AI is trained on normal telemetry behavior and detects deviations in real time. When anomaly is detected, the autonomous controller applies corrective reconfiguration logic.

---

## MVP Demonstration

Our simulation compares Human reaction delay vs AI instant response.

Results demonstrate:

- Faster anomaly detection  
- Improved system stability  
- Higher mission survival score  

The MVP is simulation-based and demonstrates how autonomous AI significantly reduces mission risk.

---

## System Architecture

Telemetry → AI Detection → Diagnosis → Autonomous Controller → System Reconfiguration  

---

## Business Model

### Target Customers:
- National space agencies  
- Satellite operators  
- Private aerospace companies  

### Revenue Model:
- Per-mission AI licensing  
- Onboard integration packages  
- Enterprise customization contracts  

AstraGuard can be deployed as a modular onboard AI layer integrated into satellite and spacecraft subsystems.

---

## How to Run

python3 -m pip install -r requirements.txt

# Run CLI simulation (backend MVP)
python3 src/main.py

# Run interactive web demo (recommended)
streamlit run app.py
