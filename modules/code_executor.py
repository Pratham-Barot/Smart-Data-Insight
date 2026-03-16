import pandas as pd
import numpy as np
import plotly.express as px
import contextlib
import io
import traceback


def execute_generated_code(code: str, df: pd.DataFrame):
    """
    Safely executes AI-generated pandas/plotly code.
    Returns: (output_text, plotly_figure, result_dataframe, error)
    """
    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "result": None,
        "fig": None
    }

    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {}, local_vars)

        output = stdout_capture.getvalue()
        fig = local_vars.get("fig", None)
        result = local_vars.get("result", None)

        return output, fig, result, None

    except Exception:
        error = traceback.format_exc()
        return "", None, None, error