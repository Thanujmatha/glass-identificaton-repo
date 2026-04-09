import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glass ID · ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #0a0c10;
    --surface:  #111318;
    --border:   #1e2230;
    --accent1:  #00d4ff;
    --accent2:  #7c3aed;
    --accent3:  #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Metric pills */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    flex: 1; min-width: 120px;
}
.metric-pill .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent1);
}
.metric-pill .lbl {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent1)22, var(--accent2)22) !important;
    color: var(--accent1) !important;
    border: 1px solid var(--accent1)44 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent1)20, var(--accent2)20) !important;
    border: 1px solid var(--accent1)66 !important;
    color: var(--accent1) !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--accent1)40, var(--accent2)40) !important;
}

/* Select/slider labels */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    font-family: 'Syne', sans-serif !important;
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

div[data-testid="stMetricValue"] {
    color: var(--accent1) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

hr { border-color: var(--border) !important; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent1);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GLASS_TYPES = {
    1: "Building Windows (Float)",
    2: "Building Windows (Non-Float)",
    3: "Vehicle Windows (Float)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

FEATURE_DESC = {
    "RI": "Refractive Index",
    "Na": "Sodium (Na)",
    "Mg": "Magnesium (Mg)",
    "Al": "Aluminum (Al)",
    "Si": "Silicon (Si)",
    "K":  "Potassium (K)",
    "Ca": "Calcium (Ca)",
    "Ba": "Barium (Ba)",
    "Fe": "Iron (Fe)"
}

PLOTLY_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e2230", zerolinecolor="#1e2230"),
    yaxis=dict(gridcolor="#1e2230", zerolinecolor="#1e2230"),
)

PALETTE = ["#00d4ff", "#7c3aed", "#f59e0b", "#10b981", "#f43f5e", "#fb923c"]

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("glass.csv")
    df["TypeLabel"] = df["Type"].map(GLASS_TYPES)
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🔬 Glass·ID</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Classifier", [
        "Random Forest", "Gradient Boosting", "SVM", "KNN", "Logistic Regression"
    ])

    st.markdown('<div class="section-title" style="margin-top:1rem">Train/Test Split</div>', unsafe_allow_html=True)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    st.markdown('<div class="section-title" style="margin-top:1rem">Feature Selection</div>', unsafe_allow_html=True)
    selected_features = st.multiselect("Features", FEATURES, default=FEATURES)
    if not selected_features:
        selected_features = FEATURES

    st.markdown('<div class="section-title" style="margin-top:1rem">Class Filter</div>', unsafe_allow_html=True)
    class_filter = st.multiselect(
        "Glass Types", list(GLASS_TYPES.values()), default=list(GLASS_TYPES.values())
    )

    st.markdown("---")
    run = st.button("▶  TRAIN MODEL", use_container_width=True)

# ── Filter Data ───────────────────────────────────────────────────────────────
filtered_df = df[df["TypeLabel"].isin(class_filter)]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Glass Identification</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">UCI Dataset · Chemical Composition Analysis · Multi-Class Classification</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Quick Stats ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Samples", len(filtered_df))
c2.metric("Features", len(selected_features))
c3.metric("Classes", filtered_df["Type"].nunique())
c4.metric("Train Set", int(len(filtered_df) * (1 - test_size)))
c5.metric("Test Set", int(len(filtered_df) * test_size))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 EDA", "🤖 Model", "🔍 Features", "🧬 PCA", "🎯 Predict"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])

    with col1:
        # Class distribution
        type_counts = filtered_df["TypeLabel"].value_counts().reset_index()
        type_counts.columns = ["Glass Type", "Count"]
        fig = px.bar(
            type_counts, x="Count", y="Glass Type", orientation="h",
            color="Count", color_continuous_scale=["#7c3aed", "#00d4ff"],
            title="Class Distribution"
        )
        fig.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                          coloraxis_showscale=False, height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Correlation heatmap
        corr = filtered_df[selected_features + ["Type"]].corr()
        fig2 = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=[[0, "#7c3aed"], [0.5, "#0a0c10"], [1, "#00d4ff"]],
            title="Correlation Matrix"
        )
        fig2.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                           height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Feature distributions
    st.markdown('<div class="section-title">Feature Distributions by Glass Type</div>', unsafe_allow_html=True)
    feat_sel = st.selectbox("Select Feature", selected_features, key="eda_feat")
    fig3 = px.violin(
        filtered_df, x="TypeLabel", y=feat_sel, color="TypeLabel",
        color_discrete_sequence=PALETTE, box=True, points="outliers",
        title=f"{FEATURE_DESC.get(feat_sel, feat_sel)} across Glass Types"
    )
    fig3.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                       showlegend=False, height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter
    st.markdown('<div class="section-title">Bivariate Scatter</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    x_feat = sc1.selectbox("X-axis", selected_features, index=0, key="sc_x")
    y_feat = sc2.selectbox("Y-axis", selected_features, index=2, key="sc_y")
    fig4 = px.scatter(
        filtered_df, x=x_feat, y=y_feat, color="TypeLabel",
        color_discrete_sequence=PALETTE, opacity=0.75,
        title=f"{x_feat} vs {y_feat}"
    )
    fig4.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                       height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Model Training & Evaluation</div>', unsafe_allow_html=True)

    X = filtered_df[selected_features].values
    y = filtered_df["Type"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    def get_model(name):
        return {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }[name]

    with st.spinner("Training…"):
        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{acc:.2%}")
    m2.metric("CV Mean", f"{cv_scores.mean():.2%}")
    m3.metric("CV Std", f"±{cv_scores.std():.3f}")
    m4.metric("Train Samples", len(X_train))

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        # Confusion matrix
        classes = sorted(filtered_df["Type"].unique())
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        class_labels = [GLASS_TYPES.get(c, str(c)) for c in classes]
        fig_cm = px.imshow(
            cm, x=class_labels, y=class_labels, text_auto=True,
            color_continuous_scale=[[0, "#0a0c10"], [1, "#00d4ff"]],
            title="Confusion Matrix"
        )
        fig_cm.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                             height=380, margin=dict(l=10, r=10, t=40, b=10))
        fig_cm.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_b:
        # CV scores bar
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_scores,
            marker=dict(
                color=cv_scores,
                colorscale=[[0, "#7c3aed"], [1, "#00d4ff"]],
                showscale=False
            )
        ))
        fig_cv.add_hline(y=cv_scores.mean(), line_dash="dot", line_color="#f59e0b",
                         annotation_text=f"Mean: {cv_scores.mean():.3f}")
        fig_cv.update_layout(**PLOTLY_THEME, title="5-Fold Cross-Validation",
                             title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                             height=380, margin=dict(l=10, r=10, t=40, b=10))
        fig_cv.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_cv, use_container_width=True)

    # Classification report
    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_rows = []
    for cls_key, vals in report.items():
        if cls_key in ("accuracy", "macro avg", "weighted avg"):
            continue
        try:
            type_int = int(cls_key)
            label = GLASS_TYPES.get(type_int, cls_key)
        except ValueError:
            label = cls_key
        report_rows.append({
            "Class": label,
            "Precision": f"{vals['precision']:.3f}",
            "Recall": f"{vals['recall']:.3f}",
            "F1-Score": f"{vals['f1-score']:.3f}",
            "Support": int(vals["support"])
        })
    st.dataframe(pd.DataFrame(report_rows).set_index("Class"),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Feature Importance & Analysis</div>', unsafe_allow_html=True)

    X2 = filtered_df[selected_features].values
    y2 = filtered_df["Type"].values
    X2_scaled = StandardScaler().fit_transform(X2)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X2_scaled, y2)
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": [FEATURE_DESC.get(f, f) for f in selected_features],
        "Importance": importances,
        "Short": selected_features
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        feat_imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#7c3aed", "#00d4ff"],
        title="Random Forest Feature Importances"
    )
    fig_fi.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                         coloraxis_showscale=False, height=380,
                         margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Box plots for top features
    top_feats = feat_imp_df.tail(4)["Short"].tolist()[::-1]
    st.markdown('<div class="section-title">Top Features — Distribution</div>', unsafe_allow_html=True)
    fig_box = make_subplots(rows=2, cols=2,
                            subplot_titles=[FEATURE_DESC.get(f, f) for f in top_feats])
    for i, feat in enumerate(top_feats):
        row, col = divmod(i, 2)
        for j, (type_id, label) in enumerate(GLASS_TYPES.items()):
            subset = filtered_df[filtered_df["Type"] == type_id][feat]
            if len(subset):
                fig_box.add_trace(go.Box(
                    y=subset, name=label, marker_color=PALETTE[j % len(PALETTE)],
                    showlegend=(i == 0), legendgroup=label
                ), row=row+1, col=col+1)
    fig_box.update_layout(**PLOTLY_THEME, height=500,
                          title_font=dict(family="Syne", size=13),
                          legend=dict(font=dict(size=9)),
                          margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_box, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PCA
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Principal Component Analysis</div>', unsafe_allow_html=True)

    X3 = filtered_df[selected_features].values
    y3 = filtered_df["Type"].values
    X3_scaled = StandardScaler().fit_transform(X3)

    n_comp = min(len(selected_features), len(filtered_df) - 1, 9)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X3_scaled)

    col_pca1, col_pca2 = st.columns(2)

    with col_pca1:
        # Scree plot
        explained = pca.explained_variance_ratio_ * 100
        cumulative = np.cumsum(explained)
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(n_comp)],
            y=explained, name="Individual",
            marker_color="#7c3aed"
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(n_comp)],
            y=cumulative, name="Cumulative",
            line=dict(color="#00d4ff", width=2), mode="lines+markers"
        ))
        fig_scree.add_hline(y=90, line_dash="dot", line_color="#f59e0b",
                            annotation_text="90%")
        fig_scree.update_layout(**PLOTLY_THEME, title="Scree Plot",
                                title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                                height=340, margin=dict(l=10, r=10, t=40, b=10),
                                yaxis_title="Variance Explained (%)",
                                legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_scree, use_container_width=True)

    with col_pca2:
        # Loadings heatmap
        loadings = pd.DataFrame(
            pca.components_[:min(6, n_comp)],
            columns=selected_features,
            index=[f"PC{i+1}" for i in range(min(6, n_comp))]
        )
        fig_load = px.imshow(
            loadings, text_auto=".2f",
            color_continuous_scale=[[0, "#7c3aed"], [0.5, "#0a0c10"], [1, "#00d4ff"]],
            title="PCA Loadings"
        )
        fig_load.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                               height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_load, use_container_width=True)

    # 2D and 3D scatter
    pca2d = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    pca2d["Type"] = [GLASS_TYPES.get(t, str(t)) for t in y3]
    fig_2d = px.scatter(
        pca2d, x="PC1", y="PC2", color="Type",
        color_discrete_sequence=PALETTE, opacity=0.8,
        title=f"PCA 2D — {explained[0]:.1f}% + {explained[1]:.1f}% variance"
    )
    fig_2d.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                         height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_2d, use_container_width=True)

    if n_comp >= 3:
        pca3d = pd.DataFrame(X_pca[:, :3], columns=["PC1", "PC2", "PC3"])
        pca3d["Type"] = [GLASS_TYPES.get(t, str(t)) for t in y3]
        fig_3d = px.scatter_3d(
            pca3d, x="PC1", y="PC2", z="PC3", color="Type",
            color_discrete_sequence=PALETTE, opacity=0.75, size_max=6,
            title="PCA 3D Projection"
        )
        fig_3d.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                             height=520, margin=dict(l=0, r=0, t=40, b=0),
                             scene=dict(
                                 bgcolor="rgba(0,0,0,0)",
                                 xaxis=dict(gridcolor="#1e2230", backgroundcolor="rgba(0,0,0,0)"),
                                 yaxis=dict(gridcolor="#1e2230", backgroundcolor="rgba(0,0,0,0)"),
                                 zaxis=dict(gridcolor="#1e2230", backgroundcolor="rgba(0,0,0,0)")
                             ))
        st.plotly_chart(fig_3d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Predict
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown("Adjust chemical composition values to predict glass type in real time.")

    # Train final model
    X_all = df[selected_features].values
    y_all = df["Type"].values
    scaler_pred = StandardScaler()
    X_all_scaled = scaler_pred.fit_transform(X_all)
    model_pred = RandomForestClassifier(n_estimators=300, random_state=42)
    model_pred.fit(X_all_scaled, y_all)

    stats = df[selected_features].describe()

    cols = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(selected_features):
        with cols[i % 3]:
            mn = float(stats[feat]["min"])
            mx = float(stats[feat]["max"])
            mean = float(stats[feat]["mean"])
            input_vals[feat] = st.slider(
                f"{FEATURE_DESC.get(feat, feat)} ({feat})",
                min_value=round(mn, 4),
                max_value=round(mx, 4),
                value=round(mean, 4),
                step=round((mx - mn) / 200, 5),
                key=f"pred_{feat}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    input_arr = np.array([[input_vals[f] for f in selected_features]])
    input_scaled = scaler_pred.transform(input_arr)
    pred_class = model_pred.predict(input_scaled)[0]
    pred_proba = model_pred.predict_proba(input_scaled)[0]
    pred_label = GLASS_TYPES.get(pred_class, f"Type {pred_class}")

    st.markdown(f"""
    <div class="card" style="border-color:#00d4ff44; text-align:center; padding:2rem;">
        <div style="font-family:'Syne',sans-serif; font-size:0.75rem; letter-spacing:4px;
                    text-transform:uppercase; color:#64748b; margin-bottom:0.5rem;">Predicted Glass Type</div>
        <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
                    background:linear-gradient(135deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            {pred_label}
        </div>
        <div style="font-family:'JetBrains Mono'; font-size:0.85rem; color:#64748b; margin-top:0.5rem;">
            Type {pred_class} · Confidence: {pred_proba.max():.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar chart
    classes_pred = model_pred.classes_
    proba_df = pd.DataFrame({
        "Glass Type": [GLASS_TYPES.get(c, f"Type {c}") for c in classes_pred],
        "Probability": pred_proba
    }).sort_values("Probability", ascending=True)

    fig_prob = px.bar(
        proba_df, x="Probability", y="Glass Type", orientation="h",
        color="Probability", color_continuous_scale=["#7c3aed", "#00d4ff"],
        title="Class Probabilities"
    )
    fig_prob.update_layout(**PLOTLY_THEME, title_font=dict(family="Syne", size=13, color="#e2e8f0"),
                           coloraxis_showscale=False, height=320,
                           margin=dict(l=10, r=10, t=40, b=10))
    fig_prob.update_xaxes(range=[0, 1])
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#334155; font-size:0.7rem; letter-spacing:2px;">'
    'UCI GLASS IDENTIFICATION DATASET · 214 SAMPLES · 9 CHEMICAL FEATURES · 6 GLASS TYPES'
    '</div>',
    unsafe_allow_html=True
)
