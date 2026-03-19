import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PlaceIQ · Placement Intelligence",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Variables ───────────────────────── */
:root {
    --bg-primary:    #080B12;
    --bg-secondary:  #0F1623;
    --bg-card:       #131C2E;
    --bg-hover:      #1A2540;
    --accent-1:      #4F8EF7;
    --accent-2:      #00D4AA;
    --accent-3:      #F7724F;
    --accent-4:      #B57BFF;
    --text-primary:  #EEF2FF;
    --text-secondary:#8A99BB;
    --border:        rgba(79,142,247,0.15);
    --shadow:        0 8px 32px rgba(0,0,0,0.4);
}

/* ── Global Reset ─────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Sidebar ──────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stSlider > div > div { background: var(--accent-1) !important; }

/* ── Metric Cards ─────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 0.55rem 1.4rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-1) !important;
    color: white !important;
}

/* ── DataFrame ────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-4)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── Alerts ───────────────────────────────── */
.stSuccess { background: rgba(0,212,170,0.1) !important; border-left: 4px solid var(--accent-2) !important; border-radius: 0 10px 10px 0 !important; }
.stError   { background: rgba(247,114,79,0.1) !important; border-left: 4px solid var(--accent-3) !important; border-radius: 0 10px 10px 0 !important; }
.stInfo    { background: rgba(79,142,247,0.1) !important; border-left: 4px solid var(--accent-1) !important; border-radius: 0 10px 10px 0 !important; }
.stWarning { background: rgba(181,123,255,0.1) !important; border-left: 4px solid var(--accent-4) !important; border-radius: 0 10px 10px 0 !important; }

/* ── Progress bar ─────────────────────────── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important; border-radius: 99px !important; }
.stProgress > div { background: var(--bg-secondary) !important; border-radius: 99px !important; }

/* ── Selectbox / Multiselect ──────────────── */
[data-baseweb="select"] > div { background: var(--bg-card) !important; border-color: var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE  (reusable)
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#8A99BB"),
    title_font=dict(family="Space Grotesk", color="#EEF2FF", size=15),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor="rgba(79,142,247,0.08)", linecolor="rgba(79,142,247,0.15)", tickfont=dict(color="#8A99BB")),
    yaxis=dict(gridcolor="rgba(79,142,247,0.08)", linecolor="rgba(79,142,247,0.15)", tickfont=dict(color="#8A99BB")),
)

PALETTE = ["#4F8EF7", "#00D4AA", "#F7724F", "#B57BFF", "#F7C84F"]

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0F1623 0%, #131C2E 60%, #1a1040 100%);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
  <div style="position:absolute;top:-60px;right:-60px;width:280px;height:280px;
              background:radial-gradient(circle, rgba(79,142,247,0.14) 0%, transparent 70%);
              border-radius:50%;pointer-events:none;"></div>
  <div style="position:absolute;bottom:-80px;left:30%;width:200px;height:200px;
              background:radial-gradient(circle, rgba(0,212,170,0.10) 0%, transparent 70%);
              border-radius:50%;pointer-events:none;"></div>

  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.6rem;">
    <span style="font-size:2rem;">🎓</span>
    <span style="
        font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;
        background:linear-gradient(90deg,#4F8EF7,#00D4AA);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-family:'Space Grotesk',sans-serif;
    ">PlaceIQ</span>
    <span style="
        background:rgba(79,142,247,0.18);border:1px solid rgba(79,142,247,0.35);
        color:#4F8EF7;font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
        padding:3px 10px;border-radius:99px;text-transform:uppercase;
        font-family:'Space Grotesk',sans-serif;
    ">ML · v2.0</span>
  </div>

  <p style="color:#A0AECF;font-size:0.95rem;margin:0;max-width:600px;line-height:1.6;font-family:'Space Grotesk',sans-serif;">
    Placement Intelligence Platform — Analyze student profiles, explore model insights,
    and predict placement outcomes powered by a trained <strong style="color:#EEF2FF;">Random Forest Classifier</strong>.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    conn = sqlite3.connect("database/placement.db")
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return pickle.load(open("models/placement_model.pkl", "rb"))

df    = load_data()
model = load_model()

# ─────────────────────────────────────────────
# SIDEBAR  – FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem;">
      <div style="font-size:1.5rem;margin-bottom:4px;">🎛️</div>
      <div style="font-weight:700;font-size:1.05rem;color:#EEF2FF;font-family:'Space Grotesk',sans-serif;">Filter Panel</div>
      <div style="color:#A0AECF;font-size:0.78rem;font-family:'Space Grotesk',sans-serif;">Refine the dataset view</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📐 Minimum CGPA**")
    min_cgpa = st.slider("", 5.0, 10.0, 5.0, step=0.1, label_visibility="collapsed")

    st.markdown("**💼 Minimum Internships**")
    min_internships = st.selectbox("", [0, 1, 2, 3], label_visibility="collapsed")

    st.markdown("**🏷️ Placement Status**")
    placement_filter = st.multiselect(
        "", ["Placed", "Not Placed"],
        default=["Placed", "Not Placed"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    total_raw = len(df)
    filtered_df = df[
        (df["cgpa"] >= min_cgpa) &
        (df["internships"] >= min_internships) &
        (df["placement"].isin(placement_filter))
    ]
    pct = round(len(filtered_df) / total_raw * 100, 1)
    st.markdown(f"""
    <div style="background:#131C2E;border:1px solid rgba(79,142,247,0.25);
                border-radius:12px;padding:1rem;text-align:center;">
        <div style="font-size:1.5rem;font-weight:700;color:#4F8EF7;font-family:'Space Grotesk',sans-serif;">{len(filtered_df)}</div>
        <div style="color:#A0AECF;font-size:0.78rem;font-family:'Space Grotesk',sans-serif;">Records ({pct}% of total)</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────
placed_count     = filtered_df[filtered_df["placement"] == "Placed"].shape[0]
not_placed_count = filtered_df[filtered_df["placement"] == "Not Placed"].shape[0]
place_rate       = round(placed_count / len(filtered_df) * 100, 1) if len(filtered_df) else 0
avg_cgpa         = round(filtered_df["cgpa"].mean(), 2) if len(filtered_df) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Total Students",  f"{len(filtered_df):,}",  delta=f"{pct}% of dataset")
c2.metric("✅ Placed",          f"{placed_count:,}",       delta=f"{place_rate}% rate")
c3.metric("❌ Not Placed",      f"{not_placed_count:,}",   delta=f"{100-place_rate}% rate", delta_color="inverse")
c4.metric("📊 Avg CGPA",        f"{avg_cgpa}",             delta="filtered view")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Analytics", "🧠  Model Insights", "🚀  Predict"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — ANALYTICS
# ═══════════════════════════════════════════════════════════
with tab1:

    # ── Data Preview ──────────────────────────
    st.markdown("#### 📋 Dataset Preview")
    st.dataframe(
        filtered_df.head(10).style.background_gradient(cmap="Blues", subset=["cgpa", "aptitude_score"]),
        use_container_width=True,
        height=340
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Placement Distribution + CGPA Histogram
    col_a, col_b = st.columns(2)

    with col_a:
        vc = filtered_df["placement"].value_counts().reset_index()
        vc.columns = ["placement", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=vc["placement"], values=vc["count"],
            hole=0.58,
            marker=dict(colors=["#4F8EF7", "#F7724F"],
                        line=dict(color="#080B12", width=3)),
            textfont=dict(family="Space Grotesk", color="white"),
        ))
        fig_pie.add_annotation(
            text=f"<b>{place_rate}%</b><br><span style='font-size:11px;'>Placed</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#EEF2FF"), align="center"
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, title="Placement Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_hist = px.histogram(
            filtered_df, x="cgpa", color="placement",
            nbins=20, barmode="overlay",
            color_discrete_sequence=["#4F8EF7", "#F7724F"],
            opacity=0.8, title="CGPA Distribution by Status"
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 2: Scatter + Internships Box ───────
    col_c, col_d = st.columns(2)

    with col_c:
        fig_sc = px.scatter(
            filtered_df, x="cgpa", y="aptitude_score",
            color="placement", size="projects",
            color_discrete_sequence=["#4F8EF7", "#F7724F"],
            opacity=0.75, title="CGPA vs Aptitude (bubble = Projects)"
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_d:
        fig_box = px.box(
            filtered_df, x="internships", y="cgpa",
            color="placement",
            color_discrete_sequence=["#4F8EF7", "#F7724F"],
            title="CGPA Spread by Internship Count"
        )
        fig_box.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation Heatmap ────────────────────
    st.markdown("#### 🔥 Correlation Matrix")
    numeric_df = filtered_df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#F7724F"],[0.5,"#131C2E"],[1,"#4F8EF7"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont=dict(size=11),
        showscale=True,
        hoverongaps=False,
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Heatmap", height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Insight strip + Download ───────────────
    avg_placed = round(df[df["placement"] == "Placed"]["cgpa"].mean(), 2)
    avg_not    = round(df[df["placement"] == "Not Placed"]["cgpa"].mean(), 2)

    i1, i2, i3 = st.columns(3)
    i1.info(f"🎯 Avg CGPA (Placed): **{avg_placed}**")
    i2.info(f"📉 Avg CGPA (Not Placed): **{avg_not}**")
    i3.info(f"📁 Rows in view: **{len(filtered_df)}**")

    st.download_button(
        "📥 Export Filtered Data (.csv)",
        filtered_df.to_csv(index=False),
        file_name="placement_filtered.csv",
        mime="text/csv",
        use_container_width=False
    )

# ═══════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════
with tab2:

    features = ["cgpa", "internships", "projects", "aptitude_score", "communication_skills"]
    X = df[features]
    y = df["placement"].map({"Not Placed": 0, "Placed": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    rf_acc  = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
    log_acc = round(accuracy_score(y_test, log_model.predict(X_test)) * 100, 2)

    # ── Header KPIs ───────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("🌲 Random Forest Accuracy", f"{rf_acc}%",  delta="Primary model")
    m2.metric("📐 Logistic Regression",    f"{log_acc}%", delta=f"{round(rf_acc - log_acc, 2)}% gap")
    m3.metric("🧪 Test Split",             "80 / 20",     delta="Stratified")

    st.markdown("<br>", unsafe_allow_html=True)

    col_e, col_f = st.columns(2)

    # ── Feature Importance ─────────────────────
    with col_e:
        importance    = model.feature_importances_
        imp_df        = pd.DataFrame({"Feature": features, "Importance": importance})
        imp_df        = imp_df.sort_values("Importance")
        imp_df["pct"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(1)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h",
            marker=dict(
                color=imp_df["Importance"],
                colorscale=[[0,"#4F8EF7"],[1,"#00D4AA"]],
                line=dict(width=0)
            ),
            text=[f"{v}%" for v in imp_df["pct"]],
            textposition="outside",
            textfont=dict(color="#EEF2FF")
        ))
        fig_imp.update_layout(**PLOTLY_LAYOUT, title="Feature Importance (Random Forest)", height=340)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Model Comparison Bar ───────────────────
    with col_f:
        fig_cmp = go.Figure()
        models_names = ["Random Forest", "Logistic Regression"]
        accs         = [rf_acc, log_acc]
        colors       = ["#4F8EF7", "#00D4AA"]

        for name, acc, clr in zip(models_names, accs, colors):
            fig_cmp.add_trace(go.Bar(
                x=[name], y=[acc], name=name,
                marker_color=clr,
                text=[f"{acc}%"], textposition="outside",
                textfont=dict(color="#EEF2FF", size=13, family="Space Grotesk"),
                width=0.4
            ))

        fig_cmp.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
            yaxis=dict(range=[0, 110], **PLOTLY_LAYOUT["yaxis"]),
            title="Model Accuracy Comparison",
            showlegend=False,
            height=340
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Confusion Matrix ───────────────────────
    st.markdown("#### 🔬 Confusion Matrix — Random Forest")
    cm     = confusion_matrix(y_test, model.predict(X_test))
    labels = ["Not Placed", "Placed"]

    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,"#131C2E"],[1,"#4F8EF7"]],
        text=cm, texttemplate="<b>%{text}</b>",
        textfont=dict(size=18, color="white"),
        showscale=False
    ))
    fig_cm.update_layout(**PLOTLY_LAYOUT, title="Predicted vs Actual", height=340,
                          xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Classification Report ──────────────────
    with st.expander("📄 Full Classification Report"):
        report = classification_report(y_test, model.predict(X_test),
                                       target_names=["Not Placed", "Placed"])
        st.code(report, language="text")

# ═══════════════════════════════════════════════════════════
# TAB 3 — PREDICTION
# ═══════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div style="background:#131C2E;border:1px solid rgba(79,142,247,0.35);
                border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.6rem;">
        <h4 style="margin:0 0 0.3rem;color:#EEF2FF;font-family:'Space Grotesk',sans-serif;font-size:1.1rem;">🚀 Placement Predictor</h4>
        <p style="margin:0;color:#A0AECF;font-size:0.88rem;font-family:'Space Grotesk',sans-serif;">
            Adjust the sliders below and click <strong style='color:#4F8EF7;'>Predict</strong> to run the model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**📐 Academic & Skills**")
        cgpa         = st.slider("CGPA",               5.0, 10.0, 7.0, 0.1)
        aptitude     = st.slider("Aptitude Score",      30,  100,  60)
        communication = st.slider("Communication Skills", 1,  10,   5)

    with col_r:
        st.markdown("**💼 Experience**")
        internships  = st.slider("Internships",  0, 3, 1)
        projects     = st.slider("Projects",     1, 5, 2)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀  Run Prediction", use_container_width=False):
        input_data = pd.DataFrame([[cgpa, internships, projects, aptitude, communication]],
                                  columns=features)

        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        prob_placed     = round(probability[1] * 100, 2)
        prob_not_placed = round(probability[0] * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.success(f"### 🎉 Likely to be Placed!")
                st.markdown(f"""
                <div style="background:#0D2218;border:1px solid rgba(0,212,170,0.35);
                            border-radius:14px;padding:1.2rem 1.6rem;margin-top:0.8rem;">
                    <div style="font-size:2.8rem;font-weight:700;color:#00D4AA;font-family:'JetBrains Mono',monospace;">
                        {prob_placed}%
                    </div>
                    <div style="color:#A0AECF;font-size:0.85rem;font-family:'Space Grotesk',sans-serif;">Placement Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"### ⚠️ Low Placement Likelihood")
                st.markdown(f"""
                <div style="background:#1E100A;border:1px solid rgba(247,114,79,0.35);
                            border-radius:14px;padding:1.2rem 1.6rem;margin-top:0.8rem;">
                    <div style="font-size:2.8rem;font-weight:700;color:#F7724F;font-family:'JetBrains Mono',monospace;">
                        {prob_not_placed}%
                    </div>
                    <div style="color:#A0AECF;font-size:0.85rem;font-family:'Space Grotesk',sans-serif;">Not-Placed Probability</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Confidence Breakdown**")
            st.progress(int(prob_placed))
            st.caption(f"Placed: {prob_placed}%  |  Not Placed: {prob_not_placed}%")

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_placed,
                number=dict(suffix="%", font=dict(size=32, color="#EEF2FF", family="Space Grotesk")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#8A99BB"),
                    bar=dict(color="#4F8EF7"),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0,  40],  color="rgba(247,114,79,0.2)"),
                        dict(range=[40, 70],  color="rgba(247,200,79,0.2)"),
                        dict(range=[70, 100], color="rgba(0,212,170,0.2)")
                    ],
                    threshold=dict(
                        line=dict(color="#00D4AA", width=3),
                        thickness=0.8, value=70
                    )
                ),
                title=dict(text="Placement Score", font=dict(size=14, color="#8A99BB"))
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk"),
                margin=dict(l=20, r=20, t=60, b=20),
                height=300
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Input Summary Card ─────────────────
        st.markdown("#### 📋 Input Summary")
        summary_df = pd.DataFrame({
            "Feature":    ["CGPA", "Internships", "Projects", "Aptitude Score", "Communication"],
            "Your Input": [cgpa, internships, projects, aptitude, communication],
            "Dataset Avg":[round(df["cgpa"].mean(),2), round(df["internships"].mean(),2),
                           round(df["projects"].mean(),2), round(df["aptitude_score"].mean(),2),
                           round(df["communication_skills"].mean(),2)]
        })
        summary_df["vs Avg"] = (
            (summary_df["Your Input"] - summary_df["Dataset Avg"]) /
            summary_df["Dataset Avg"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
