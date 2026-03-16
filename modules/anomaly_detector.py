import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """
    Runs multiple anomaly detection methods on numeric columns.
    Returns original df with anomaly flags + a summary report.
    """

    result = {}

    # --- Extract numeric columns only ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        result["error"] = "No numeric columns found for anomaly detection."
        return result

    # --- Prepare clean data (fill nulls with median) ---
    df_clean = df[numeric_cols].copy()
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # --- Step 4.2: Normalize with StandardScaler ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # ================================
    # Method 1 — IsolationForest (ML)
    # ================================
    iso_model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    iso_preds = iso_model.fit_predict(X_scaled)
    iso_scores = -iso_model.score_samples(X_scaled)

    # ================================
    # Method 2 — Z-Score (Statistical)
    # ================================
    z_scores = np.abs(stats.zscore(df_clean, nan_policy="omit"))
    z_anomaly = pd.Series((z_scores > 3).any(axis=1), index=df.index)

    # ================================
    # Method 3 — IQR (Statistical)
    # ================================
    iqr_anomaly = pd.Series(False, index=df.index)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        iqr_anomaly |= (df_clean[col] < lower) | (df_clean[col] > upper)

    # --- Step 4.5: Merge flags into result DataFrame ---
    df_result = df.copy()
    df_result["anomaly_isolation_forest"] = iso_preds == -1
    df_result["anomaly_zscore"] = z_anomaly.values if hasattr(z_anomaly, 'values') else z_anomaly
    df_result["anomaly_iqr"]              = iqr_anomaly.values
    df_result["anomaly_score"]            = iso_scores.round(4)

    # Combined flag — anomaly if detected by ANY method
    df_result["is_anomaly"] = (
        df_result["anomaly_isolation_forest"] |
        df_result["anomaly_zscore"] |
        df_result["anomaly_iqr"]
    )

    # --- Summary Report ---
    total = len(df_result)
    n_iso  = int(df_result["anomaly_isolation_forest"].sum())
    n_z    = int(df_result["anomaly_zscore"].sum())
    n_iqr  = int(df_result["anomaly_iqr"].sum())
    n_combined = int(df_result["is_anomaly"].sum())

    result["df_with_anomalies"] = df_result
    result["numeric_cols"]      = numeric_cols
    result["summary"] = {
        "total_rows"              : total,
        "isolation_forest_count"  : n_iso,
        "isolation_forest_percent": round(n_iso / total * 100, 2),
        "zscore_count"            : n_z,
        "zscore_percent"          : round(n_z / total * 100, 2),
        "iqr_count"               : n_iqr,
        "iqr_percent"             : round(n_iqr / total * 100, 2),
        "combined_count"          : n_combined,
        "combined_percent"        : round(n_combined / total * 100, 2),
    }

    return result