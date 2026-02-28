# AstraGuard Technical Documentation

## 1. System Overview

AstraGuard is a simulation-based autonomous AI system designed to detect and mitigate spacecraft subsystem failures in real time. The system reduces dependency on Earth-based control by enabling onboard anomaly detection and automated response.

---

## 2. Architecture

The system follows this pipeline:

Telemetry Simulation → AI Anomaly Detection → Failure Diagnosis → Autonomous Controller → System Reconfiguration

Telemetry data is generated through a simulated spacecraft environment.

---

## 3. Telemetry Simulation

The telemetry simulator generates synthetic spacecraft data including:

- Battery level  
- Temperature  
- Signal strength  
- CPU load  

Random noise is added to simulate realistic fluctuations. Failure injection mechanisms simulate subsystem degradation scenarios.

---

## 4. AI Anomaly Detection

The system uses Isolation Forest (unsupervised anomaly detection).

The model is trained on normal telemetry behavior and detects deviations in real time.

Isolation Forest was chosen because:

- It performs well on high-dimensional anomaly detection
- It does not require labeled failure data
- It is computationally lightweight for onboard use

---

## 5. Autonomous Response Logic

When anomaly is detected:

- Power anomaly → Non-critical systems shutdown  
- Thermal anomaly → Cooling mode activation  
- Communication anomaly → Backup channel switch  

This rule-based controller simulates onboard automatic reconfiguration.

---

## 6. AI vs Human Comparison

The system simulates two scenarios:

1. Human-controlled spacecraft (reaction delay simulated)
2. AI-controlled spacecraft (instant detection and response)

Metrics compared:

- Remaining battery percentage  
- System stability duration  
- Mission survival score  

---

## 7. Technology Stack

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 8. Limitations

- Simulation-based environment  
- Simplified subsystem modeling  
- Rule-based recovery logic  

Future versions may integrate physics-based validation models and reinforcement learning for advanced optimization.

---

## 9. Future Work

- Integration with real satellite telemetry datasets  
- Reinforcement learning-based decision engine  
- Physics-informed anomaly validation  
- Cloud-based mission monitoring dashboard  

---

End of Technical Documentation.
