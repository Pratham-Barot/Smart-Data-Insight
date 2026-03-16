import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from modules.data_profiler import profile_dataset
from modules.chart_engine import (
    generate_histograms,
    generate_correlation_heatmap,
    generate_bar_charts,
    generate_boxplots,
    generate_datetime_trend,
    generate_scatter_matrix,
    generate_pie_charts
)
from modules.anomaly_detector import detect_anomalies
from modules.gemini_agent import ask_gemini, get_auto_summary, extract_code
from modules.code_executor import execute_generated_code
from utils.session_memory import (
    init_session_state,
    reset_on_new_file,
    append_chat,
    clear_chat
)
import os

load_dotenv()

GEMINI_MODEL = "models/gemini-2.5-flash"

st.set_page_config(
    page_title="SmartDataInsight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 SmartDataInsight")
st.caption("AI-Powered Data Analytics Agent — Powered by Google Gemini 2.5 Flash")

# --- Initialize Session State ---
init_session_state()

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")

# API Status Indicator
st.sidebar.markdown("### API Status")
try:
    from google import genai
    test_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    st.sidebar.success("Gemini API Connected")
except Exception:
    st.sidebar.error("Gemini API Not Connected")

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json"],
    help="Supported formats: CSV, Excel, JSON"
)

# Sidebar project info
st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.info(
    "SmartDataInsight is an AI-powered data analytics agent "
    "that automatically profiles datasets, detects anomalies, "
    "generates charts, and answers questions in plain English."
)
st.sidebar.markdown("**Built with:**")
st.sidebar.markdown(
    "- Google Gemini 2.5 Flash\n"
    "- Streamlit\n"
    "- Pandas + Plotly\n"
    "- Scikit-learn"
)

# --- Load File ---
def load_file(file) -> pd.DataFrame:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif file.name.endswith(".json"):
            df = pd.read_json(file)
        else:
            st.error("Unsupported file format.")
            return None

        # Validate file is not empty
        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid dataset.")
            return None

        # Validate minimum size
        if df.shape[1] < 2:
            st.error("Dataset must have at least 2 columns.")
            return None

        return df

    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")
        return None

# --- Load on Upload ---
if uploaded_file:
    if st.session_state.filename != uploaded_file.name:
        df = load_file(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.profile = profile_dataset(df)
            st.session_state.filename = uploaded_file.name

            # Reset all previous analysis
            reset_on_new_file()

            # Auto generate Gemini summary on upload
            with st.spinner("Gemini is analyzing your dataset..."):
                df_sample_json = df.head(5).to_json(orient="records")
                st.session_state.auto_summary = get_auto_summary(
                    st.session_state.profile,
                    df_sample_json
                )
            st.sidebar.success(f"✓ {uploaded_file.name}")

# --- Main Display ---
if st.session_state.df is not None:
    df = st.session_state.df
    profile = st.session_state.profile

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Profile",
        "📈 EDA Charts",
        "🚨 Anomaly Detection",
        "🤖 AI Chat",
        "📄 Export Report"
    ])

    # =====================
    # TAB 1 — DATA PROFILE
    # =====================
    with tab1:
        st.subheader("Dataset Overview")

        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", profile["shape"]["rows"])
        col2.metric("Columns", profile["shape"]["columns"])
        col3.metric("Duplicate Rows", profile["duplicate_rows"])
        col4.metric("Numeric Cols", len(profile["numeric_columns"]))
        col5.metric("Categorical Cols", len(profile["categorical_columns"]))

        # --- AI First Look Summary ---
        if st.session_state.auto_summary:
            st.markdown("#### Gemini First Look Summary")
            st.info(st.session_state.auto_summary)

        st.divider()

        # Column roles table
        st.subheader("Column Roles")
        roles_data = []
        for col, role in profile["column_roles"].items():
            null_pct = profile["null_percent"].get(col, 0)
            cardinality = profile["cardinality"].get(col, 0)
            roles_data.append({
                "Column": col,
                "Role": role,
                "Null %": f"{null_pct}%",
                "Unique Values": cardinality,
                "Data Type": profile["dtypes"].get(col, "")
            })
        st.dataframe(
            pd.DataFrame(roles_data),
            use_container_width=True
        )

        st.divider()

        # Numeric statistics
        if profile["numeric_stats"]:
            st.subheader("Numeric Statistics")
            st.dataframe(
                pd.DataFrame(profile["numeric_stats"]).T,
                use_container_width=True
            )

        st.divider()

        # Top values for categorical columns
        if profile["categorical_columns"]:
            st.subheader("Top Values — Categorical Columns")
            cat_cols = profile["categorical_columns"]
            cols = st.columns(min(3, len(cat_cols)))
            for i, col in enumerate(cat_cols[:6]):
                with cols[i % 3]:
                    st.write(f"**{col}**")
                    top_vals = profile["top_values"].get(col, {})
                    st.dataframe(
                        pd.DataFrame(
                            list(top_vals.items()),
                            columns=["Value", "Count"]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

        st.divider()

        # Raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df, use_container_width=True)

    # =====================
    # TAB 2 — EDA CHARTS
    # =====================
    with tab2:
        st.subheader("Exploratory Data Analysis")

        numeric_cols = profile["numeric_columns"]
        categorical_cols = profile["categorical_columns"]
        datetime_cols = profile["datetime_columns"]

        if not numeric_cols and not categorical_cols:
            st.warning("No numeric or categorical columns found for charting.")
        else:
            # --- Datetime Trend ---
            if datetime_cols:
                st.markdown("#### Time Series Trend")
                trend_fig = generate_datetime_trend(
                    df, datetime_cols[0], numeric_cols
                )
                if trend_fig:
                    st.plotly_chart(trend_fig, use_container_width=True)

            # --- Histograms ---
            if numeric_cols:
                st.markdown("#### Distributions (Histograms)")
                hist_charts = generate_histograms(df, numeric_cols)
                cols_layout = st.columns(2)
                for i, (col_name, fig) in enumerate(hist_charts):
                    with cols_layout[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)

            # --- Boxplots ---
            if numeric_cols:
                st.markdown("#### Outlier View (Boxplots)")
                box_charts = generate_boxplots(df, numeric_cols)
                cols_layout = st.columns(2)
                for i, (col_name, fig) in enumerate(box_charts):
                    with cols_layout[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)

            # --- Correlation Heatmap ---
            if len(numeric_cols) >= 2:
                st.markdown("#### Correlation Heatmap")
                heatmap = generate_correlation_heatmap(df, numeric_cols)
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)

            # --- Scatter Matrix ---
            if len(numeric_cols) >= 2:
                st.markdown("#### Scatter Matrix")
                scatter = generate_scatter_matrix(df, numeric_cols)
                if scatter:
                    st.plotly_chart(scatter, use_container_width=True)

            # --- Bar Charts ---
            if categorical_cols:
                st.markdown("#### Categorical Column Counts")
                bar_charts = generate_bar_charts(df, categorical_cols)
                cols_layout = st.columns(2)
                for i, (col_name, fig) in enumerate(bar_charts):
                    with cols_layout[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)

            # --- Pie Charts ---
            if categorical_cols:
                st.markdown("#### Categorical Distributions (Pie)")
                pie_charts = generate_pie_charts(df, categorical_cols)
                if pie_charts:
                    cols_layout = st.columns(2)
                    for i, (col_name, fig) in enumerate(pie_charts):
                        with cols_layout[i % 2]:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No low-cardinality categorical columns found.")

    # ===========================
    # TAB 3 — ANOMALY DETECTION
    # ===========================
    with tab3:
        st.subheader("ML-Powered Anomaly Detection")
        st.caption(
            "Uses 3 methods: IsolationForest (ML) + Z-Score + IQR. "
            "A row is flagged if ANY method detects it as an anomaly."
        )

        numeric_cols = profile["numeric_columns"]

        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            st.sidebar.divider()
            st.sidebar.markdown("### Anomaly Settings")
            contamination = st.sidebar.slider(
                "Sensitivity (contamination)",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Higher = more anomalies detected"
            )

            run_btn = st.button(
                "Run Anomaly Detection",
                type="primary",
                use_container_width=True
            )

            if run_btn:
                with st.spinner("Running anomaly detection..."):
                    st.session_state.anomaly_result = detect_anomalies(
                        df, contamination
                    )

            if st.session_state.anomaly_result:
                res = st.session_state.anomaly_result

                if "error" in res:
                    st.error(res["error"])
                else:
                    summary = res["summary"]
                    df_anomaly = res["df_with_anomalies"]

                    # Summary Metrics
                    st.markdown("#### Detection Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Rows", summary["total_rows"])
                    c2.metric(
                        "IsolationForest",
                        summary["isolation_forest_count"],
                        f"{summary['isolation_forest_percent']}%"
                    )
                    c3.metric(
                        "Z-Score",
                        summary["zscore_count"],
                        f"{summary['zscore_percent']}%"
                    )
                    c4.metric(
                        "Combined Anomalies",
                        summary["combined_count"],
                        f"{summary['combined_percent']}%"
                    )

                    st.divider()

                    # Scatter Plot
                    st.markdown("#### Anomaly Scatter Plot")
                    num_cols = res["numeric_cols"]
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x_axis = st.selectbox("X Axis", num_cols, index=0)
                    with col_y:
                        y_axis = st.selectbox(
                            "Y Axis", num_cols,
                            index=min(1, len(num_cols) - 1)
                        )

                    fig_scatter = px.scatter(
                        df_anomaly,
                        x=x_axis, y=y_axis,
                        color="is_anomaly",
                        color_discrete_map={
                            True: "#EF553B",
                            False: "#636EFA"
                        },
                        title=f"Anomalies — {x_axis} vs {y_axis}",
                        hover_data=["anomaly_score"],
                        labels={"is_anomaly": "Anomaly"}
                    )
                    fig_scatter.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    st.divider()

                    # Anomaly Score Distribution
                    st.markdown("#### Anomaly Score Distribution")
                    fig_hist = px.histogram(
                        df_anomaly,
                        x="anomaly_score",
                        color="is_anomaly",
                        color_discrete_map={
                            True: "#EF553B",
                            False: "#636EFA"
                        },
                        title="Distribution of Anomaly Scores",
                        nbins=40,
                        labels={"is_anomaly": "Anomaly"}
                    )
                    fig_hist.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.divider()

                    # Method Comparison
                    st.markdown("#### Method Comparison")
                    method_df = pd.DataFrame({
                        "Method": [
                            "IsolationForest",
                            "Z-Score",
                            "IQR",
                            "Combined"
                        ],
                        "Anomalies Detected": [
                            summary["isolation_forest_count"],
                            summary["zscore_count"],
                            summary["iqr_count"],
                            summary["combined_count"]
                        ]
                    })
                    fig_bar = px.bar(
                        method_df,
                        x="Method",
                        y="Anomalies Detected",
                        title="Anomalies Detected per Method",
                        color="Method",
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        text="Anomalies Detected"
                    )
                    fig_bar.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.divider()

                    # Anomalous Rows Table
                    st.markdown("#### Anomalous Rows")
                    anomalous_rows = df_anomaly[
                        df_anomaly["is_anomaly"]
                    ].sort_values("anomaly_score", ascending=False)
                    st.dataframe(
                        anomalous_rows,
                        use_container_width=True
                    )
                    st.download_button(
                        label="Download Anomalous Rows as CSV",
                        data=anomalous_rows.to_csv(index=False),
                        file_name="anomalous_rows.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Click 'Run Anomaly Detection' to start.")

    # =====================
    # TAB 4 — AI CHAT
    # =====================
    with tab4:
        st.subheader("Ask Your Data Anything")
        st.caption(
            "Gemini will analyze your data, answer questions, "
            "generate Python code, and run it live."
        )

        # --- Suggested Questions ---
        st.markdown("#### Try These Questions")
        suggestions = [
            "What are the top 5 insights?",
            "Which columns are most correlated?",
            "Show a chart of the most important column",
            "What data quality issues exist?",
            "Summarize key statistics in plain English",
        ]
        cols_suggest = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols_suggest[i]:
                if st.button(
                    suggestion,
                    key=f"suggest_{i}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = suggestion

        st.divider()

        # --- Render Chat History ---
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["parts"][0]["text"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["parts"][0]["text"])

        if not st.session_state.chat_history:
            st.info(
                "Ask any question about your data below. "
                "Gemini will answer with insights, code, and charts."
            )

        # --- Chat Input ---
        user_question = st.chat_input(
            "e.g. Which product line has the highest sales?"
        )

        # Handle suggested question button click
        if st.session_state.pending_question:
            user_question = st.session_state.pending_question
            st.session_state.pending_question = None

        # --- Process Question ---
        if user_question:

            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    df_sample_json = df.head(10).to_json(orient="records")
                    response = ask_gemini(
                        user_question,
                        profile,
                        df_sample_json,
                        st.session_state.chat_history
                    )

                # Show Gemini response text
                st.markdown(response)

                # Extract and run any generated code
                code = extract_code(response)
                if code:
                    st.markdown("**Generated Code:**")
                    st.code(code, language="python")

                    with st.spinner("Running generated code..."):
                        output, fig, result, error = execute_generated_code(
                            code, df
                        )

                    if error:
                        st.error(f"Code error: {error}")
                    else:
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        if output:
                            st.text(output)
                        if result is not None:
                            try:
                                st.dataframe(result, use_container_width=True)
                            except Exception:
                                st.write(result)

            # Save to chat history using session memory
            append_chat("user", user_question)
            append_chat("model", response)

        # --- Clear Chat Button ---
        if st.session_state.chat_history:
            if st.button("Clear Chat History", type="secondary"):
                clear_chat()
                st.rerun()

    # ========================
    # TAB 5 — EXPORT REPORT
    # ========================
    with tab5:
        st.subheader("AI-Generated Analysis Report")
        st.caption(
            "Gemini will write a complete professional data analysis "
            "report based on your dataset and chat history."
        )

        # --- Report Config ---
        st.markdown("#### Report Settings")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            report_tone = st.selectbox(
                "Report Tone",
                ["Business Executive", "Technical Data Scientist",
                 "Simple & Non-Technical"],
                index=0
            )
        with col_r2:
            report_sections = st.multiselect(
                "Include Sections",
                ["Dataset Overview", "Data Quality",
                 "Key Insights", "Anomaly Summary",
                 "Recommendations", "Next Steps"],
                default=["Dataset Overview", "Data Quality",
                         "Key Insights", "Recommendations"]
            )

        st.divider()

        # --- Generate Button ---
        gen_report_btn = st.button(
            "Generate Full Report",
            type="primary",
            use_container_width=True
        )

        if gen_report_btn:
            with st.spinner("Gemini is writing your report..."):

                # Build anomaly summary for report
                anomaly_summary_text = "Anomaly detection not run yet."
                if st.session_state.anomaly_result:
                    s = st.session_state.anomaly_result["summary"]
                    anomaly_summary_text = (
                        f"Total anomalies detected: "
                        f"{s['combined_count']} "
                        f"({s['combined_percent']}% of data). "
                        f"IsolationForest: {s['isolation_forest_count']}, "
                        f"Z-Score: {s['zscore_count']}, "
                        f"IQR: {s['iqr_count']}."
                    )

                # Build chat summary for report
                chat_summary = "No chat questions asked yet."
                if st.session_state.chat_history:
                    questions = [
                        msg["parts"][0]["text"]
                        for msg in st.session_state.chat_history
                        if msg["role"] == "user"
                    ]
                    chat_summary = (
                        "User investigated these topics: "
                        + ", ".join(questions[:5])
                    )

                report_prompt = f"""
Write a professional data analysis report with the following settings:

Tone: {report_tone}
Sections to include: {", ".join(report_sections)}

Dataset Information:
- Filename: {st.session_state.filename}
- Rows: {profile["shape"]["rows"]}
- Columns: {profile["shape"]["columns"]}
- Numeric columns: {", ".join(profile["numeric_columns"])}
- Categorical columns: {", ".join(profile["categorical_columns"])}
- Duplicate rows: {profile["duplicate_rows"]}
- Null percentages: {profile["null_percent"]}

Numeric Statistics:
{profile.get("numeric_stats", {})}

Anomaly Detection Results:
{anomaly_summary_text}

User Investigation Summary:
{chat_summary}

Auto Summary Previously Generated:
{st.session_state.auto_summary}

Instructions:
- Write in clear structured markdown format
- Use headers for each section
- Include specific numbers and column names
- Make recommendations actionable and specific
- End with a concise Executive Summary
- Minimum 500 words
"""
                df_sample_json = df.head(5).to_json(orient="records")
                st.session_state.report_text = ask_gemini(
                    report_prompt,
                    profile,
                    df_sample_json
                )

        # --- Display Report ---
        if st.session_state.report_text:
            st.markdown("#### Generated Report")
            st.markdown(st.session_state.report_text)

            st.divider()

            # --- Download Options ---
            st.markdown("#### Download Report")
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.download_button(
                    label="Download as Markdown (.md)",
                    data=st.session_state.report_text,
                    file_name=f"report_{st.session_state.filename}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col_d2:
                plain_text = st.session_state.report_text.replace(
                    "#", ""
                ).replace("**", "").replace("*", "")
                st.download_button(
                    label="Download as Text (.txt)",
                    data=plain_text,
                    file_name=f"report_{st.session_state.filename}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            st.success(
                "Report generated successfully! "
                "Download it using the buttons above."
            )
        else:
            st.info(
                "Configure your report settings above "
                "and click 'Generate Full Report'."
            )

else:
    st.info("👈 Upload a dataset from the sidebar to begin.")
    st.markdown("""
    ### What SmartDataInsight can do for you:
    - **Auto-profile** your dataset instantly
    - **Detect anomalies** using ML (IsolationForest)
    - **Generate charts** automatically
    - **Answer questions** about your data in plain English
    - **Export** a full AI-written analysis report

    ### Supported formats:
    CSV, Excel (.xlsx, .xls), JSON
    """)