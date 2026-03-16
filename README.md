# Smart-Data-Insight — AI-Powered Data Analytics Agent

An intelligent data analytics agent built with Google Gemini 2.5 Flash
and Streamlit that automatically analyzes any dataset and answers
questions in plain English.

## Features

- Auto-profiles any CSV, Excel, or JSON dataset instantly
- Detects anomalies using 3 ML methods (IsolationForest, Z-Score, IQR)
- Generates EDA charts automatically (histograms, heatmaps, scatter plots)
- Answers natural language questions about your data
- Generates and executes Python code live inside the app
- Exports a full AI-written professional analysis report

## Tech Stack

- Google Gemini 2.5 Flash — AI reasoning and code generation
- Streamlit — Web application framework
- Pandas — Data manipulation
- Plotly — Interactive charts
- Scikit-learn — Anomaly detection (IsolationForest)
- SciPy — Statistical anomaly detection (Z-Score)

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/yourusername/Smart-Data-Insight.git
cd Smart-Data-Insight

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your API key
Create a .env file in the root folder and add this line:
GEMINI_API_KEY=your_gemini_api_key_here

Get your free API key at: https://aistudio.google.com

### 5. Run the app
streamlit run app.py

## Project Structure

Smart-Data-Insight/

app.py — Main Streamlit application

requirements.txt — Project dependencies

.env — API key (never commit this to git)

README.md — Project documentation

.gitignore — Git ignore rules

modules/

    data_profiler.py — Auto dataset profiling

    anomaly_detector.py — ML anomaly detection (IsolationForest, Z-Score, IQR)

    chart_engine.py — Auto chart generation (histograms, heatmaps, scatter plots)

    gemini_agent.py — Gemini AI agent (reasoning, code generation, narration)

    code_executor.py — Safe Python code executor (sandboxed exec)

    __init__.py — Module initializer

utils/

    session_memory.py — Streamlit session state manager

    __init__.py — Module initializer

## Usage

1. Upload any CSV, Excel or JSON dataset from the sidebar
2. View auto-generated profile and Gemini summary on Tab 1
3. Explore auto-generated charts on Tab 2
4. Run anomaly detection on Tab 3
5. Ask questions in plain English on Tab 4
6. Generate and download full report on Tab 5

## Supported File Formats

CSV, Excel (.xlsx, .xls), JSON

## 🙋 Author

Built by **Pratham** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/pratham-barot-1a66b62a6/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Pratham-Barot)