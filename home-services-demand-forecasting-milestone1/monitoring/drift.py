"""Simple drift-monitoring skeleton."""
import pandas as pd
from scipy.stats import ks_2samp

def ks_drift_alert(reference: pd.Series, current: pd.Series, threshold=0.05):
    """Return whether a feature distribution has statistically significant drift."""
    stat, p_value = ks_2samp(reference.dropna(), current.dropna())
    return {"drift": bool(p_value < threshold), "ks_stat": float(stat), "p_value": float(p_value)}

if __name__ == "__main__":
    print("Drift monitoring skeleton ready. Supply reference/current feature data.")
