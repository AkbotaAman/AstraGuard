from __future__ import annotations
import pandas as pd


def diagnose_failure(row: pd.Series) -> str:
    """
    Very simple heuristic diagnosis based on current telemetry values.
    """
    if row["temperature"] > 50:
        return "thermal"
    if row["battery"] < 35:
        return "power"
    if row["signal"] < 40:
        return "comm"
    if row["cpu_load"] > 85:
        return "cpu"
    return "unknown"


def apply_recovery(row: pd.Series, failure: str) -> pd.Series:
    """
    Apply an immediate "reconfiguration" effect (simulated) for next-step stability.
    This modifies a copy of the row to reflect mitigation.
    """
    r = row.copy()

    if failure == "thermal":
        # cooling mode: reduce temp growth + reduce cpu
        r["temperature"] = max(20.0, r["temperature"] - 3.0)
        r["cpu_load"] = max(0.0, r["cpu_load"] - 8.0)
        r["battery"] = max(0.0, r["battery"] - 0.3)  # cooling costs a bit
    elif failure == "power":
        # shut down noncritical systems: reduce cpu usage, slow drain
        r["cpu_load"] = max(0.0, r["cpu_load"] - 10.0)
        r["battery"] = min(100.0, r["battery"] + 0.6)  # stabilize power bus
    elif failure == "comm":
        # switch to backup antenna: increase signal, small cpu increase
        r["signal"] = min(100.0, r["signal"] + 12.0)
        r["cpu_load"] = min(100.0, r["cpu_load"] + 2.0)
        r["battery"] = max(0.0, r["battery"] - 0.2)
    elif failure == "cpu":
        # reset noncritical processes
        r["cpu_load"] = max(0.0, r["cpu_load"] - 15.0)
        r["battery"] = max(0.0, r["battery"] - 0.2)

    return r


def mission_damage(row: pd.Series) -> float:
    """
    Simple "damage" metric: higher means worse system state.
    """
    damage = 0.0
    # battery low is bad
    if row["battery"] < 40:
        damage += (40 - row["battery"]) * 0.6
    # too hot is bad
    if row["temperature"] > 45:
        damage += (row["temperature"] - 45) * 1.2
    # comm loss is bad
    if row["signal"] < 50:
        damage += (50 - row["signal"]) * 0.7
    # cpu overload is bad
    if row["cpu_load"] > 80:
        damage += (row["cpu_load"] - 80) * 0.4
    return damage
