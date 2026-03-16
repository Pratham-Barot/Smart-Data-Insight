import pandas as pd
import numpy as np

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Automatically profiles a DataFrame and returns
    a comprehensive dictionary of statistics and metadata.
    """

    profile = {}

    # --- Basic Shape ---
    profile["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }

    # --- Column Names ---
    profile["columns"] = list(df.columns)

    # --- Data Types ---
    profile["dtypes"] = df.dtypes.astype(str).to_dict()

    # --- Null Analysis ---
    profile["null_counts"] = df.isnull().sum().to_dict()
    profile["null_percent"] = (
        df.isnull().mean() * 100
    ).round(2).to_dict()

    # --- Duplicate Rows ---
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # --- Cardinality (unique values per column) ---
    profile["cardinality"] = {
        col: int(df[col].nunique()) for col in df.columns
    }

    # --- Numeric Statistics ---
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        profile["numeric_stats"] = (
            numeric_df.describe().round(2).to_dict()
        )
        profile["numeric_columns"] = list(numeric_df.columns)
    else:
        profile["numeric_stats"] = {}
        profile["numeric_columns"] = []

    # --- Categorical Columns ---
    cat_df = df.select_dtypes(include=["object", "category"])
    profile["categorical_columns"] = list(cat_df.columns)

    # --- Datetime Columns ---
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == object:
            try:
                pd.to_datetime(
                    df[col].dropna().head(10),
                    format="mixed",
                    dayfirst=False
                )
                datetime_cols.append(col)
            except Exception:
                pass
    profile["datetime_columns"] = datetime_cols

    # --- Column Roles ---
    profile["column_roles"] = {}
    for col in df.columns:
        if col in datetime_cols:
            profile["column_roles"][col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 20:
                profile["column_roles"][col] = "numeric_categorical"
            else:
                profile["column_roles"][col] = "numeric"
        elif df[col].dtype == object:
            if df[col].nunique() <= 20:
                profile["column_roles"][col] = "categorical"
            else:
                profile["column_roles"][col] = "text"
        else:
            profile["column_roles"][col] = "other"

    # --- Top Values for Categorical Columns ---
    profile["top_values"] = {}
    for col in profile["categorical_columns"]:
        profile["top_values"][col] = (
            df[col].value_counts().head(5).to_dict()
        )

    # --- Sample Rows ---
    profile["sample_rows"] = df.head(5).to_dict(orient="records")

    # --- Memory Usage ---
    profile["memory_usage_kb"] = round(
        df.memory_usage(deep=True).sum() / 1024, 2
    )

    return profile