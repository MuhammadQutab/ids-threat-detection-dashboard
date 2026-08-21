import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Geo-Aware ML Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# PREMIUM CYBER SECURITY THEME
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0,255,255,0.12), transparent 22%),
            radial-gradient(circle at top right, rgba(0,255,128,0.10), transparent 20%),
            radial-gradient(circle at bottom left, rgba(59,130,246,0.10), transparent 25%),
            linear-gradient(135deg, #020617 0%, #031426 38%, #061A2E 70%, #020617 100%);
        color: #F8FAFC;
    }

    html, body, [class*="css"] {
        color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #010812 0%, #041225 100%);
        border-right: 1px solid rgba(34,211,238,0.35);
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    .main-title {
        font-size: 44px;
        font-weight: 900;
        color: #67E8F9;
        text-shadow: 0 0 12px rgba(103,232,249,0.55), 0 0 28px rgba(0,255,255,0.25);
        margin-bottom: 6px;
    }

    .sub-title {
        font-size: 18px;
        color: #D1FAE5;
        margin-bottom: 26px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 850;
        color: #F8FAFC;
        margin-top: 18px;
        margin-bottom: 12px;
        text-shadow: 0 0 10px rgba(34,211,238,0.30);
    }

    .overview-box {
        background: rgba(5, 18, 34, 0.96);
        border-left: 5px solid #22D3EE;
        border-radius: 16px;
        padding: 20px;
        color: #F8FAFC;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(34,211,238,0.08);
    }

    .metric-card {
        background: linear-gradient(160deg, rgba(8,20,40,0.98), rgba(3,10,22,0.98));
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(103,232,249,0.28);
        box-shadow: 0 0 24px rgba(0,255,255,0.09);
        text-align: center;
        min-height: 120px;
    }

    .metric-icon {
        font-size: 23px;
        margin-bottom: 7px;
    }

    .metric-number {
        font-size: 31px;
        font-weight: 900;
        color: #22D3EE;
        text-shadow: 0 0 12px rgba(34,211,238,0.35);
    }

    .metric-label {
        font-size: 14px;
        color: #E2E8F0;
    }

    .threat-banner {
        background: linear-gradient(90deg, rgba(239,68,68,0.22), rgba(251,146,60,0.13), rgba(34,211,238,0.10));
        border: 1px solid rgba(248,113,113,0.45);
        color: #F8FAFC;
        padding: 18px 20px;
        border-radius: 16px;
        font-weight: 800;
        box-shadow: 0 0 22px rgba(239,68,68,0.12);
        margin-bottom: 20px;
    }

    .console-box {
        background: #020B16;
        border: 1px solid rgba(0,255,153,0.35);
        border-radius: 16px;
        box-shadow: 0 0 24px rgba(0,255,153,0.12);
        overflow: hidden;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .console-header {
        background: linear-gradient(90deg, #031826 0%, #06324D 100%);
        padding: 11px 17px;
        color: #BAF7FF;
        font-weight: 850;
        border-bottom: 1px solid rgba(0,255,153,0.22);
        font-size: 14px;
        letter-spacing: 0.6px;
    }

    .console-content {
        padding: 17px;
        color: #86EFAC;
        font-family: Consolas, Monaco, monospace;
        font-size: 14px;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .glass-card {
        background: rgba(7, 20, 38, 0.94);
        border: 1px solid rgba(0,255,255,0.20);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 0 18px rgba(0,255,255,0.08);
        margin-bottom: 18px;
        color: #F8FAFC;
    }

    /* Main action buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #06283D 0%, #0E7490 45%, #115E59 100%) !important;
        color: #F8FAFC !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        border: 1px solid rgba(103,232,249,0.55) !important;
        border-radius: 13px !important;
        padding: 14px 18px !important;
        box-shadow: 0 0 18px rgba(34,211,238,0.28) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0891B2 0%, #0E7490 50%, #0F766E 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #A5F3FC !important;
        box-shadow: 0 0 28px rgba(34,211,238,0.42) !important;
        transform: translateY(-1px);
    }

    /* Download button fixed visibility */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #111827 0%, #0F172A 100%) !important;
        color: #F8FAFC !important;
        font-weight: 850 !important;
        border: 1px solid rgba(34,211,238,0.55) !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        box-shadow: 0 0 16px rgba(34,211,238,0.20) !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(90deg, #0E7490 0%, #115E59 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #A5F3FC !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #E0F2FE !important;
        font-weight: 750 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #67E8F9 !important;
        border-bottom-color: #22D3EE !important;
    }

    /* Dataframes and labels */
    label, .stMarkdown, .stText, p, span {
        color: #F8FAFC;
    }

    .small-note {
        color: #CBD5E1;
        font-size: 13px;

    }

            /* Dataframe table readability */
    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
    }

    div[data-testid="stDataFrame"] * {
        color: #111827 !important;
    }

    div[data-testid="stDataFrame"] input {
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }

    div[data-testid="stDataFrame"] button {
        color: #111827 !important;
        background-color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_xgboost_ids_model.pkl")
    label_encoder = joblib.load("final_label_encoder.pkl")
    feature_columns = joblib.load("final_feature_columns.pkl")
    return model, label_encoder, feature_columns

def assign_severity(prediction):
    severity_map = {
        "Benign": "Low",
        "Botnet": "Critical",
        "BruteForce": "High",
        "DDoS": "Critical",
        "DoS": "High"
    }
    return severity_map.get(prediction, "Medium")

def assign_geo_context(prediction, index):
    geo_pool = [
        {"Source IP": "185.220.101.45", "Country": "Germany", "Region": "Berlin", "Latitude": 52.5200, "Longitude": 13.4050},
        {"Source IP": "45.155.205.19", "Country": "Netherlands", "Region": "Amsterdam", "Latitude": 52.3676, "Longitude": 4.9041},
        {"Source IP": "91.219.236.21", "Country": "Russia", "Region": "Moscow", "Latitude": 55.7558, "Longitude": 37.6173},
        {"Source IP": "103.48.119.82", "Country": "Singapore", "Region": "Central", "Latitude": 1.3521, "Longitude": 103.8198},
        {"Source IP": "172.67.88.30", "Country": "United States", "Region": "California", "Latitude": 36.7783, "Longitude": -119.4179},
        {"Source IP": "185.199.109.44", "Country": "United Kingdom", "Region": "London", "Latitude": 51.5072, "Longitude": -0.1276}
    ]

    if prediction == "Benign":
        return {
            "Source IP": "192.168.1.10",
            "Country": "Internal Network",
            "Region": "Local",
            "Latitude": 0.0,
            "Longitude": 0.0
        }

    return geo_pool[index % len(geo_pool)]

try:
    model, label_encoder, feature_columns = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("## 🛡️ IDS Control Panel")
st.sidebar.markdown("**Model:** XGBoost Classifier")
st.sidebar.markdown("**Dataset:** CSE-CIC-IDS2018")
st.sidebar.markdown("**Model Type:** Multi-class ML intrusion detector")
st.sidebar.markdown("**Classes:**")
st.sidebar.markdown("""
- Benign  
- DDoS  
- DoS  
- Botnet  
- BruteForce  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Input Mode")
mode = st.sidebar.radio(
    "Choose input mode",
    ["Use sample_test_data_100.csv", "Upload new CSV"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Geo-location is used only as alert enrichment after prediction. "
    "It is not used as a model training feature."
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Geo-Aware ML Intrusion Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Machine Learning + Cyber Security Prototype for Intrusion Detection, Threat Analysis and Geo-Aware Alert Enrichment</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">📚</div>
        <div class="metric-number">5</div>
        <div class="metric-label">Traffic Classes</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🧾</div>
        <div class="metric-number">77</div>
        <div class="metric-label">Network Flow Features</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🤖</div>
        <div class="metric-number">XGB</div>
        <div class="metric-label">Selected Final Model</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    status_text = "READY" if model_loaded else "ERROR"
    status_color = "#22C55E" if model_loaded else "#EF4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-number" style="color:{status_color};">{status_text}</div>
        <div class="metric-label">Model Status</div>
    </div>
    """, unsafe_allow_html=True)

if not model_loaded:
    st.error("Model files could not be loaded.")
    st.code(model_error)
    st.stop()

# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------
st.markdown('<div class="section-title">📌 Project Overview</div>', unsafe_allow_html=True)
st.markdown("""
<div class="overview-box">
This dashboard demonstrates a <b>machine learning-based cyber intrusion detection prototype</b> developed using the
<b>CSE-CIC-IDS2018 dataset</b>. The selected final model is <b>XGBoost</b>, trained to classify network traffic into
five categories: <b>Benign, DDoS, DoS, Botnet and BruteForce</b>.<br><br>

The prototype connects model prediction with a practical cyber-security workflow: traffic input, attack classification,
confidence scoring, severity mapping, geo-aware enrichment, alert analytics and downloadable results.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA INPUT
# ---------------------------------------------------------
st.markdown('<div class="section-title">📂 Network Traffic Input</div>', unsafe_allow_html=True)

if mode == "Use sample_test_data_100.csv":
    if os.path.exists("sample_test_data_100.csv"):
        data = pd.read_csv("sample_test_data_100.csv")
        st.success("Default sample_test_data_100.csv loaded successfully.")
    else:
        st.error("sample_test_data_100.csv not found in the dashboard folder.")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing the required 77 network traffic feature columns",
        type=["csv"]
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV file loaded successfully.")
    else:
        st.warning("Please upload a compatible CSV file to continue.")
        st.stop()

missing_features = [col for col in feature_columns if col not in data.columns]
if missing_features:
    st.error("The input file is missing required model feature columns.")
    st.write(missing_features)
    st.stop()

X_input = data[feature_columns].astype("float32")
actual_available = "Actual_Class" in data.columns

tabs = st.tabs([
    "📄 Data Preview",
    "🧪 Single Prediction",
    "🚀 Bulk Detection",
    "🧠 Research Notes"
])

# ---------------------------------------------------------
# TAB 1 DATA PREVIEW
# ---------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-title">📄 Uploaded / Sample Dataset Preview</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p1.metric("Rows Loaded", len(data))
    p2.metric("Feature Columns Required", len(feature_columns))
    p3.metric("Actual Labels Available", "Yes" if actual_available else "No")

    st.dataframe(
    data.head(12).style.set_properties(**{
        "background-color": "#FFFFFF",
        "color": "#111827"
    }),
    use_container_width=True
    )

# ---------------------------------------------------------
# TAB 2 SINGLE PREDICTION
# ---------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-title">🧪 Single-Row Live Prediction Mode</div>', unsafe_allow_html=True)

    row_index = st.number_input(
        "Select a record index for single-row inspection",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1
    )

    if st.button("🔍 ANALYSE SELECTED RECORD"):
        single_row = X_input.iloc[[row_index]]
        single_pred = model.predict(single_row)
        single_proba = model.predict_proba(single_row)

        pred_label = label_encoder.inverse_transform(single_pred)[0]
        confidence = float(np.max(single_proba))
        severity = assign_severity(pred_label)
        geo = assign_geo_context(pred_label, int(row_index))

        actual_text = "Not available"
        if actual_available:
            actual_text = str(data.loc[row_index, "Actual_Class"])

        console_text = f"""
[ RECORD INDEX ]        {row_index}
[ ACTUAL CLASS ]        {actual_text}
[ PREDICTED CLASS ]     {pred_label}
[ CONFIDENCE ]          {confidence * 100:.2f}%
[ SEVERITY ]            {severity}
[ SOURCE IP ]           {geo['Source IP']}
[ COUNTRY ]             {geo['Country']}
[ REGION ]              {geo['Region']}
[ GEO NOTE ]            Contextual enrichment only
"""

        st.markdown("""
        <div class="threat-banner">
        ⚡ Single record analysed successfully. Review the SOC console output below.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="console-box">
            <div class="console-header">SINGLE RECORD ANALYSIS CONSOLE</div>
            <div class="console-content">{console_text}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3 BULK DETECTION
# ---------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-title">🚀 Bulk Intrusion Detection Engine</div>', unsafe_allow_html=True)

    if st.button("🚀 RUN INTRUSION DETECTION PREDICTION"):
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)

        predicted_labels = label_encoder.inverse_transform(predictions)
        confidence_scores = probabilities.max(axis=1)

        results = pd.DataFrame({
            "Predicted Class": predicted_labels,
            "Confidence Score": np.round(confidence_scores, 4)
        })

        if actual_available:
            results.insert(0, "Actual Class", data["Actual_Class"])
            results["Correct Prediction"] = results["Actual Class"] == results["Predicted Class"]

        results["Severity"] = results["Predicted Class"].apply(assign_severity)

        geo_records = [
            assign_geo_context(prediction, i)
            for i, prediction in enumerate(results["Predicted Class"])
        ]
        geo_df = pd.DataFrame(geo_records)
        results = pd.concat([results, geo_df], axis=1)

        total_records = len(results)
        malicious_records = len(results[results["Predicted Class"] != "Benign"])
        benign_records = len(results[results["Predicted Class"] == "Benign"])
        avg_confidence = results["Confidence Score"].mean()
        dominant_class = results["Predicted Class"].value_counts().idxmax()

        threat_state = "ATTACK TRAFFIC DETECTED" if malicious_records > 0 else "NO MAJOR ALERTS"

        st.markdown(f"""
        <div class="threat-banner">
        🚨 Threat State: {threat_state} | Malicious Alerts: {malicious_records} | Average Confidence: {avg_confidence * 100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        console_text = f"""
[ SYSTEM STATUS ]      XGBoost IDS Engine ............ ONLINE
[ TOTAL RECORDS ]      {total_records}
[ MALICIOUS ALERTS ]   {malicious_records}
[ BENIGN RECORDS ]     {benign_records}
[ AVG CONFIDENCE ]     {avg_confidence * 100:.2f}%
[ DOMINANT CLASS ]     {dominant_class}
[ THREAT STATE ]       {threat_state}
[ GEO ENRICHMENT ]     ENABLED
[ OUTPUT STATUS ]      Prediction table generated
"""

        st.markdown('<div class="section-title">💻 SOC Live Console</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="console-box">
            <div class="console-header">SECURITY OPERATIONS CONSOLE</div>
            <div class="console-content">{console_text}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Detection Summary</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📁</div>
                <div class="metric-number">{total_records}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🚨</div>
                <div class="metric-number">{malicious_records}</div>
                <div class="metric-label">Malicious Alerts</div>
            </div>
            """, unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-number">{benign_records}</div>
                <div class="metric-label">Benign Records</div>
            </div>
            """, unsafe_allow_html=True)

        with s4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-number">{avg_confidence * 100:.2f}%</div>
                <div class="metric-label">Average Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        if actual_available:
            sample_accuracy = results["Correct Prediction"].mean() * 100
            st.success(f"Sample Accuracy on Current Input: {sample_accuracy:.2f}%")

        st.markdown('<div class="section-title">📈 Threat Analytics</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        attack_counts = results["Predicted Class"].value_counts().reset_index()
        attack_counts.columns = ["Class", "Count"]

        fig_bar = px.bar(
            attack_counts,
            x="Class",
            y="Count",
            title="Predicted Traffic Class Distribution",
            text="Count",
            color="Class",
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig_bar.update_layout(
            paper_bgcolor="#071426",
            plot_bgcolor="#071426",
            font_color="#F8FAFC",
            title_font_size=18
        )

        severity_counts = results["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]

        fig_pie = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            title="Alert Severity Breakdown",
            hole=0.45,
            color="Severity",
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#FACC15",
                "High": "#FB923C",
                "Critical": "#EF4444"
            }
        )
        fig_pie.update_layout(
            paper_bgcolor="#071426",
            font_color="#F8FAFC",
            title_font_size=18
        )

        with chart_col1:
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-title">🚨 Prediction Results with Geo-Aware Alert Enrichment</div>', unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ DOWNLOAD PREDICTION RESULTS CSV",
            data=csv_data,
            file_name="ids_prediction_results.csv",
            mime="text/csv"
        )

        st.markdown('<div class="section-title">🔥 High-Risk Alerts</div>', unsafe_allow_html=True)
        high_risk = results[results["Severity"].isin(["High", "Critical"])]

        if len(high_risk) > 0:
            st.warning(f"{len(high_risk)} high-risk or critical alerts detected.")
            st.dataframe(
                high_risk[["Predicted Class", "Confidence Score", "Severity", "Source IP", "Country", "Region"]].head(20),
                use_container_width=True
            )
        else:
            st.success("No high-risk alerts detected.")

        st.markdown('<div class="section-title">🌍 Geo-Aware Alert Map</div>', unsafe_allow_html=True)
        malicious_geo = results[results["Predicted Class"] != "Benign"].copy()

        if len(malicious_geo) > 0:
            fig_map = px.scatter_geo(
                malicious_geo,
                lat="Latitude",
                lon="Longitude",
                color="Severity",
                hover_name="Predicted Class",
                hover_data=["Country", "Region", "Source IP", "Confidence Score"],
                title="Malicious Alert Geo-Enrichment View",
                color_discrete_map={
                    "High": "#FB923C",
                    "Critical": "#EF4444"
                }
            )
            fig_map.update_layout(
                paper_bgcolor="#071426",
                font_color="#F8FAFC",
                geo=dict(
                    bgcolor="#071426",
                    showland=True,
                    landcolor="#0F172A",
                    showocean=True,
                    oceancolor="#031525",
                    showcountries=True
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No malicious geo-alerts available to plot.")

        if actual_available:
            st.markdown('<div class="section-title">🧠 Model Evaluation on Current Input</div>', unsafe_allow_html=True)

            y_true_labels = results["Actual Class"]
            y_pred_labels = results["Predicted Class"]

            cm = confusion_matrix(
                y_true_labels,
                y_pred_labels,
                labels=list(label_encoder.classes_)
            )

            cm_fig = px.imshow(
                cm,
                text_auto=True,
                x=label_encoder.classes_,
                y=label_encoder.classes_,
                color_continuous_scale="Tealgrn",
                title="Confusion Matrix"
            )
            cm_fig.update_layout(
                paper_bgcolor="#071426",
                plot_bgcolor="#071426",
                font_color="#F8FAFC"
            )
            st.plotly_chart(cm_fig, use_container_width=True)

            report_dict = classification_report(
                y_true_labels,
                y_pred_labels,
                labels=list(label_encoder.classes_),
                output_dict=True,
                zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.markdown("### Classification Report")
            st.dataframe(report_df, use_container_width=True)

    else:
        st.markdown("""
        <div class="glass-card">
            <h3>⚡ Awaiting Bulk Prediction</h3>
            <p>Click the prediction button above to activate the IDS engine and generate cyber threat analytics.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4 RESEARCH NOTES
# ---------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-title">🧠 Research and Prototype Notes</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="overview-box">
    <b>Research Positioning:</b><br>
    This dashboard is a proof-of-concept prototype. The model does not represent a complete commercial IDS.
    It demonstrates how a trained machine learning model can be integrated into a practical intrusion monitoring workflow.
    <br><br>

    <b>Model Scope:</b><br>
    The current model classifies five categories: Benign, DDoS, DoS, Botnet and BruteForce.
    Web Attack and Infiltration files are not included in this version and may be added as future work.
    <br><br>

    <b>Geo-Aware Enrichment:</b><br>
    Geo-location is added after prediction as contextual alert enrichment. It is not used as a training feature and should
    not be interpreted as confirmed attacker identity or exact attacker location because real attackers may use VPNs,
    proxies or compromised machines.
    <br><br>

    <b>Practical Contribution:</b><br>
    The novelty of this prototype is the combination of model comparison, selected XGBoost deployment, confidence scoring,
    severity mapping, dashboard-based alert presentation and geo-aware contextual enrichment.
    </div>
    """, unsafe_allow_html=True)