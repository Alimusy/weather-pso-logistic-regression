import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Prediction | PSO + LR",
    page_icon="🌦️",
    layout="wide"
)

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "artifacts"
    return {
        "feature_names":     joblib.load(f"{base}/feature_names.pkl"),
        "selected_names":    joblib.load(f"{base}/selected_names.pkl"),
        "selected_features": joblib.load(f"{base}/selected_features.pkl"),
        "base_metrics":      joblib.load(f"{base}/base_metrics.pkl"),
        "pso_metrics":       joblib.load(f"{base}/pso_metrics.pkl"),
        "base_cm":           joblib.load(f"{base}/base_cm.pkl"),
        "pso_cm":            joblib.load(f"{base}/pso_cm.pkl"),
        "cost_history":      joblib.load(f"{base}/cost_history.pkl"),
        "baseline_pipeline": joblib.load(f"{base}/baseline_lr_pipeline.pkl"),
        "pso_pipeline":      joblib.load(f"{base}/pso_lr_pipeline.pkl"),
        "scaler":            joblib.load(f"{base}/scaler.pkl"),
    }

data = load_artifacts()

# ── Constants ─────────────────────────────────────────────────
METRIC_KEYS   = ["acc", "f1", "prec", "rec", "auc"]
METRIC_LABELS = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC"]

WIND_DIRS = {
    "E":0,"ENE":1,"ESE":2,"N":3,"NE":4,"NNE":5,
    "NNW":6,"NW":7,"S":8,"SE":9,"SSE":10,"SSW":11,
    "SW":12,"Unknown":13,"W":14,"WNW":15,"WSW":16
}
MONTHS = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July":7,"August":8,
    "September":9,"October":10,"November":11,"December":12
}

# ── Shared helpers ────────────────────────────────────────────
def metric_cards(metrics_dict, color):
    cols = st.columns(5)
    for col, key, label in zip(cols, METRIC_KEYS, METRIC_LABELS):
        val = np.mean(metrics_dict[key])
        std = np.std(metrics_dict[key])
        col.markdown(f"""
        <div style="background:{color};border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:13px;color:#e0e0e0;margin-bottom:4px;">{label}</div>
            <div style="font-size:28px;font-weight:700;color:#ffffff;">{val:.4f}</div>
            <div style="font-size:11px;color:#cccccc;">± {std:.4f}</div>
        </div>""", unsafe_allow_html=True)

def cm_chart(cm, title, color):
    labels = ["No Rain","Rain"]
    z_text = [[str(v) for v in row] for row in cm.tolist()]
    fig = ff.create_annotated_heatmap(
        z=cm.tolist(), x=labels, y=labels,
        annotation_text=z_text,
        colorscale=[[0,"#1a1a2e"],[1,color]],
        showscale=False
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15,color="#ffffff")),
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        margin=dict(t=50,b=40,l=60,r=20), height=320
    )
    return fig

def prediction_badge(label, prob):
    if label == 1:
        color, icon, text = "#1b5e20", "🌧️", "RAIN EXPECTED"
    else:
        color, icon, text = "#1a237e", "☀️", "NO RAIN EXPECTED"
    st.markdown(f"""
    <div style="background:{color};border-radius:12px;padding:24px;text-align:center;margin-top:16px;">
        <div style="font-size:40px;">{icon}</div>
        <div style="font-size:22px;font-weight:700;color:#ffffff;margin:8px 0;">{text}</div>
        <div style="font-size:14px;color:#cccccc;">Confidence: {prob*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:10px 0 20px;">
    <div style="font-size:32px;">🌦️</div>
    <div style="font-size:18px;font-weight:700;color:#ffffff;">Weather Prediction</div>
    <div style="font-size:12px;color:#aaaaaa;">PSO Feature Selection + LR</div>
</div>""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Baseline Logistic Regression","🔬 PSO + Logistic Regression"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size:12px;color:#aaaaaa;">
    <b>Dataset:</b> Rain in Australia<br>
    <b>Total Features:</b> {len(data['feature_names'])}<br>
    <b>PSO Selected:</b> {len(data['selected_names'])}<br>
    <b>CV Folds:</b> 5-Fold Stratified<br>
    <b>Sampling:</b> SMOTE
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 1 — BASELINE LR
# ════════════════════════════════════════════════════════════════
if page == "📊 Baseline Logistic Regression":

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a237e,#283593);
                border-radius:12px;padding:28px 32px;margin-bottom:28px;">
        <h1 style="color:#ffffff;margin:0;font-size:28px;">📊 Baseline Logistic Regression</h1>
        <p style="color:#90caf9;margin:8px 0 0;font-size:15px;">
            Standard LR trained on all 22 features — 5-Fold CV with SMOTE
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📈 Performance Metrics (Mean ± Std)")
    metric_cards(data["base_metrics"], "#1a237e")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 🔲 Confusion Matrix")
        st.plotly_chart(cm_chart(data["base_cm"],"Baseline LR — Confusion Matrix","#1565c0"), use_container_width=True)
    with col2:
        st.markdown("#### 🧩 Features Used (All 22)")
        html = "".join([
            f'<div style="display:flex;align-items:center;padding:5px 10px;margin:3px 0;'
            f'border-radius:6px;background:#1e2a3a;">'
            f'<span style="color:#90caf9;font-size:12px;width:24px;">{i}.</span>'
            f'<span style="color:#ffffff;font-size:13px;">{f}</span></div>'
            for i, f in enumerate(data["feature_names"], 1)
        ])
        st.markdown(f'<div style="max-height:340px;overflow-y:auto;">{html}</div>', unsafe_allow_html=True)

    # Bar chart
    vals = [np.mean(data["base_metrics"][k]) for k in METRIC_KEYS]
    stds = [np.std(data["base_metrics"][k]) for k in METRIC_KEYS]
    fig = go.Figure(go.Bar(
        x=METRIC_LABELS, y=vals,
        error_y=dict(type='data', array=stds, visible=True, color="#90caf9"),
        marker_color=["#1565c0","#1976d2","#1e88e5","#2196f3","#42a5f5"],
        text=[f"{v:.4f}" for v in vals], textposition="outside",
        textfont=dict(color="#ffffff", size=12)
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#ffffff"),
        yaxis=dict(range=[0,1.1], gridcolor="#1e2a3a"),
        xaxis=dict(gridcolor="#1e2a3a"),
        margin=dict(t=20,b=20), height=320
    )
    st.markdown("#### 📊 Metric Overview")
    st.plotly_chart(fig, use_container_width=True)

    # ── Prediction Form ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Make a Prediction (All 22 Features)")

    with st.form("baseline_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            month    = st.selectbox("Month", list(MONTHS.keys()))
            min_temp = st.number_input("Min Temp (°C)", -10.0, 50.0, 12.0)
            max_temp = st.number_input("Max Temp (°C)", -10.0, 55.0, 24.0)
            temp9am  = st.number_input("Temp 9am (°C)", -10.0, 55.0, 15.0)
            temp3pm  = st.number_input("Temp 3pm (°C)", -10.0, 55.0, 22.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 0.0)
            evap     = st.number_input("Evaporation (mm)", 0.0, 150.0, 5.0)
            sunshine = st.number_input("Sunshine (hrs)", 0.0, 14.0, 7.0)
        with c2:
            wind_gust_dir = st.selectbox("Wind Gust Direction", list(WIND_DIRS.keys()))
            wind_gust_spd = st.number_input("Wind Gust Speed (km/h)", 0.0, 150.0, 40.0)
            wind_dir_9am  = st.selectbox("Wind Dir 9am", list(WIND_DIRS.keys()))
            wind_dir_3pm  = st.selectbox("Wind Dir 3pm", list(WIND_DIRS.keys()))
            wind_spd_9am  = st.number_input("Wind Speed 9am (km/h)", 0.0, 130.0, 15.0)
            wind_spd_3pm  = st.number_input("Wind Speed 3pm (km/h)", 0.0, 130.0, 20.0)
        with c3:
            hum_9am  = st.number_input("Humidity 9am (%)", 0.0, 100.0, 70.0)
            hum_3pm  = st.number_input("Humidity 3pm (%)", 0.0, 100.0, 50.0)
            pres_9am = st.number_input("Pressure 9am (hPa)", 970.0, 1045.0, 1015.0)
            pres_3pm = st.number_input("Pressure 3pm (hPa)", 970.0, 1045.0, 1012.0)
            cloud_9am = st.slider("Cloud 9am (oktas)", 0, 9, 4)
            cloud_3pm = st.slider("Cloud 3pm (oktas)", 0, 9, 4)
            rain_today = st.selectbox("Rain Today?", ["No","Yes"])

        submitted = st.form_submit_button("🌦️ Predict", use_container_width=True)

    if submitted:
        m = MONTHS[month]
        month_sin = np.sin(2 * np.pi * m / 12)
        month_cos = np.cos(2 * np.pi * m / 12)

        features = np.array([[
            min_temp, max_temp, rainfall, evap, sunshine,
            WIND_DIRS[wind_gust_dir], wind_gust_spd,
            WIND_DIRS[wind_dir_9am], WIND_DIRS[wind_dir_3pm],
            wind_spd_9am, wind_spd_3pm,
            hum_9am, hum_3pm, pres_9am, pres_3pm,
            cloud_9am, cloud_3pm, temp9am, temp3pm,
            1 if rain_today == "Yes" else 0,
            month_sin, month_cos
        ]])

        scaled   = data["scaler"].transform(features)
        pred     = data["baseline_pipeline"].predict(scaled)[0]
        prob     = data["baseline_pipeline"].predict_proba(scaled)[0][pred]
        prediction_badge(pred, prob)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — PSO + LR
# ════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);
                border-radius:12px;padding:28px 32px;margin-bottom:28px;">
        <h1 style="color:#ffffff;margin:0;font-size:28px;">🔬 PSO + Logistic Regression</h1>
        <p style="color:#a5d6a7;margin:8px 0 0;font-size:15px;">
            LR on PSO-selected features — 30 particles · 100 iterations · 5-Fold CV with SMOTE
        </p>
    </div>""", unsafe_allow_html=True)

    total     = len(data["feature_names"])
    selected  = len(data["selected_names"])
    dropped   = total - selected
    reduction = (dropped / total) * 100

    c1, c2, c3 = st.columns(3)
    for col, val, label, color in [
        (c1, total,              "Total Features", "#1565c0"),
        (c2, selected,           "PSO Selected",   "#2e7d32"),
        (c3, f"{reduction:.1f}%","Reduction",      "#6a1b9a"),
    ]:
        col.markdown(f"""
        <div style="background:{color};border-radius:10px;padding:16px;text-align:center;">
            <div style="font-size:13px;color:#e0e0e0;">{label}</div>
            <div style="font-size:32px;font-weight:700;color:#ffffff;">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Performance Metrics (Mean ± Std)")
    metric_cards(data["pso_metrics"], "#1b5e20")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 🔲 Confusion Matrix")
        st.plotly_chart(cm_chart(data["pso_cm"],"PSO + LR — Confusion Matrix","#2e7d32"), use_container_width=True)
    with col2:
        st.markdown(f"#### 🧩 PSO Selected Features ({selected})")
        dropped_names = [f for f in data["feature_names"] if f not in data["selected_names"]]
        sel_html = "".join([
            f'<div style="display:flex;align-items:center;padding:5px 10px;margin:3px 0;'
            f'border-radius:6px;background:#1b2e1b;">'
            f'<span style="color:#a5d6a7;font-size:12px;margin-right:8px;">✅</span>'
            f'<span style="color:#ffffff;font-size:13px;">{f}</span></div>'
            for f in data["selected_names"]
        ])
        drop_html = "".join([
            f'<div style="display:flex;align-items:center;padding:5px 10px;margin:3px 0;'
            f'border-radius:6px;background:#2a1a1a;">'
            f'<span style="color:#ef9a9a;font-size:12px;margin-right:8px;">❌</span>'
            f'<span style="color:#aaaaaa;font-size:13px;text-decoration:line-through;">{f}</span></div>'
            for f in dropped_names
        ])
        st.markdown("**Kept:**")
        st.markdown(f'<div style="max-height:160px;overflow-y:auto;">{sel_html}</div>', unsafe_allow_html=True)
        st.markdown("**Dropped:**")
        st.markdown(f'<div>{drop_html}</div>', unsafe_allow_html=True)

    # Convergence curve
    st.markdown("#### 📉 PSO Convergence Curve")
    cost = data["cost_history"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(cost)+1)), y=cost,
        mode="lines", line=dict(color="#66bb6a", width=2.5),
        fill="tozeroy", fillcolor="rgba(102,187,106,0.1)"
    ))
    fig2.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#ffffff"),
        xaxis=dict(title="Iteration", gridcolor="#1e2a3a"),
        yaxis=dict(title="Best Cost (1 − F1)", gridcolor="#1e2a3a"),
        margin=dict(t=10,b=40), height=280, showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Comparison bar
    st.markdown("#### ⚖️ Baseline LR vs PSO + LR")
    base_vals = [np.mean(data["base_metrics"][k]) for k in METRIC_KEYS]
    pso_vals  = [np.mean(data["pso_metrics"][k])  for k in METRIC_KEYS]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Baseline LR", x=METRIC_LABELS, y=base_vals,
        marker_color="#1565c0",
        text=[f"{v:.4f}" for v in base_vals], textposition="outside",
        textfont=dict(color="#ffffff", size=11)))
    fig3.add_trace(go.Bar(name="PSO + LR", x=METRIC_LABELS, y=pso_vals,
        marker_color="#2e7d32",
        text=[f"{v:.4f}" for v in pso_vals], textposition="outside",
        textfont=dict(color="#ffffff", size=11)))
    fig3.update_layout(
        barmode="group",
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#ffffff"),
        yaxis=dict(range=[0,1.1], gridcolor="#1e2a3a"),
        xaxis=dict(gridcolor="#1e2a3a"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#333", borderwidth=1),
        margin=dict(t=10,b=20), height=320
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Prediction Form (PSO features only) ──────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Make a Prediction (PSO Selected Features Only)")

    with st.form("pso_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            month         = st.selectbox("Month", list(MONTHS.keys()))
            min_temp      = st.number_input("Min Temp (°C)", -10.0, 50.0, 12.0)
            temp9am       = st.number_input("Temp 9am (°C)", -10.0, 55.0, 15.0)
            rainfall      = st.number_input("Rainfall (mm)", 0.0, 400.0, 0.0)
            evap          = st.number_input("Evaporation (mm)", 0.0, 150.0, 5.0)
            sunshine      = st.number_input("Sunshine (hrs)", 0.0, 14.0, 7.0)
        with c2:
            wind_gust_dir = st.selectbox("Wind Gust Direction", list(WIND_DIRS.keys()))
            wind_gust_spd = st.number_input("Wind Gust Speed (km/h)", 0.0, 150.0, 40.0)
            wind_spd_3pm  = st.number_input("Wind Speed 3pm (km/h)", 0.0, 130.0, 20.0)
            hum_9am       = st.number_input("Humidity 9am (%)", 0.0, 100.0, 70.0)
            hum_3pm       = st.number_input("Humidity 3pm (%)", 0.0, 100.0, 50.0)
        with c3:
            pres_9am  = st.number_input("Pressure 9am (hPa)", 970.0, 1045.0, 1015.0)
            pres_3pm  = st.number_input("Pressure 3pm (hPa)", 970.0, 1045.0, 1012.0)
            cloud_9am = st.slider("Cloud 9am (oktas)", 0, 9, 4)
            cloud_3pm = st.slider("Cloud 3pm (oktas)", 0, 9, 4)

        submitted = st.form_submit_button("🌦️ Predict", use_container_width=True)

    if submitted:
        m = MONTHS[month]
        month_sin = np.sin(2 * np.pi * m / 12)

        # Full 22-feature array (in correct order), then slice PSO features
        all_features = np.array([[
            min_temp, 0, rainfall, evap, sunshine,
            WIND_DIRS[wind_gust_dir], wind_gust_spd,
            0, 0, 0, wind_spd_3pm,
            hum_9am, hum_3pm, pres_9am, pres_3pm,
            cloud_9am, cloud_3pm, temp9am, 0,
            0, month_sin, 0
        ]])

        scaled_all  = data["scaler"].transform(all_features)
        scaled_pso  = scaled_all[:, data["selected_features"]]
        pred        = data["pso_pipeline"].predict(scaled_pso)[0]
        prob        = data["pso_pipeline"].predict_proba(scaled_pso)[0][pred]
        prediction_badge(pred, prob)
