"""
app.py — Streamlit frontend for the Binary Pattern Classifier.

Run with:  streamlit run app.py
"""

import json
import pathlib

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent
MODEL_PATH = BASE / "model_weights.pth"
HISTORY_PATH = BASE / "training_history.json"


# ── Model (mirrors train.py) ────────────────────────────────────────────────
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Binary Pattern Classifier",
    page_icon="🧠",
    layout="wide",
)

# ── Inject dark-mode / glassmorphism CSS ─────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import Google Font ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global dark background & typography ───────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(145deg, #0a0a1a 0%, #0f1128 40%, #13072e 100%);
    color: #e0e0f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: rgba(15, 15, 40, 0.85);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(120, 100, 255, 0.15);
}

/* ── Glass card mixin ──────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(100, 80, 255, 0.18);
}

/* ── Metric card ───────────────────────────────────────────────────────── */
.metric-card {
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #9090c0;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #7c3aed, #6d28d9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-card .value.green {
    background: linear-gradient(135deg, #34d399, #10b981, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .value.rose {
    background: linear-gradient(135deg, #fb7185, #e11d48, #be123c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Section headers ───────────────────────────────────────────────────── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: #c4b5fd;
    margin: 2rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(167, 139, 250, 0.4), transparent);
}

/* ── Weight badges ─────────────────────────────────────────────────────── */
.weight-badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 10px;
    padding: 0.5rem 1.1rem;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 0.95rem;
    color: #c4b5fd;
    margin: 0.25rem;
}

/* ── Radio buttons ─────────────────────────────────────────────────────── */
div[data-testid="stRadio"] label {
    color: #c4b5fd !important;
    font-weight: 600;
}

/* ── Plotly chart containers ───────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 16px;
    overflow: hidden;
}

/* ── Hide Streamlit branding ───────────────────────────────────────────── */
#MainMenu, footer, header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helper: render a glass metric card ───────────────────────────────────
def metric_card(label: str, value: str, color_class: str = "") -> None:
    st.markdown(
        f"""
        <div class="glass-card metric-card">
            <div class="label">{label}</div>
            <div class="value {color_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon: str, text: str) -> None:
    st.markdown(
        f'<div class="section-header">{icon}&ensp;{text}</div>',
        unsafe_allow_html=True,
    )


# ── Load model (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model() -> Perceptron:
    model = Perceptron()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
    else:
        st.error(
            "⚠ Model weights not found. Run `python train.py` first.",
            icon="🚨",
        )
        st.stop()
    return model


@st.cache_data
def load_history() -> dict:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {"epochs": [], "loss": []}


model = load_model()
history = load_history()

# Extract learned parameters
w1 = model.linear.weight.data[0, 0].item()
w2 = model.linear.weight.data[0, 1].item()
bias = model.linear.bias.data[0].item()

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 50%, #4f46e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        ">🧠 Binary Pattern Classifier</h1>
        <p style="color: #7c7ca8; font-size: 0.95rem; letter-spacing: 1px;">
            Single-Layer Perceptron · OR Gate · PyTorch
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: Input controls ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1.5rem;">
            <span style="font-size:2.2rem;">⚡</span>
            <h3 style="color:#c4b5fd; margin:0.3rem 0 0 0; font-weight:700;">
                Input Controls
            </h3>
            <p style="color:#7c7ca8; font-size:0.8rem; margin-top:0.2rem;">
                Select binary inputs for the OR gate
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    x1 = st.radio("**Input X₁**", options=[0, 1], horizontal=True, index=0)
    x2 = st.radio("**Input X₂**", options=[0, 1], horizontal=True, index=0)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding:0.8rem; border-radius:12px;
                    background: rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.2);">
            <div style="color:#9090c0; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:0.3rem;">Truth Table</div>
            <code style="color:#c4b5fd; font-size:0.85rem; line-height:1.8;">
                0 OR 0 → 0<br>
                0 OR 1 → 1<br>
                1 OR 0 → 1<br>
                1 OR 1 → 1
            </code>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Inference ────────────────────────────────────────────────────────────
with torch.no_grad():
    inp = torch.tensor([[float(x1), float(x2)]])
    prob = model(inp).item()
    predicted_class = int(prob >= 0.5)

# ── Row 1: Prediction metrics ───────────────────────────────────────────
section_header("🎯", "Live Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    metric_card(
        "Predicted Class",
        str(predicted_class),
        "green" if predicted_class == 1 else "rose",
    )
with col2:
    metric_card("Sigmoid Probability", f"{prob:.4f}")
with col3:
    expected = int(x1 or x2)
    match = predicted_class == expected
    metric_card(
        "Correct?",
        "✓ Yes" if match else "✗ No",
        "green" if match else "rose",
    )

# ── Row 2: Learned parameters ───────────────────────────────────────────
section_header("⚙️", "Learned Parameters")

st.markdown(
    f"""
    <div class="glass-card" style="display:flex; justify-content:center;
         gap:1rem; flex-wrap:wrap; text-align:center;">
        <span class="weight-badge">w₁ = {w1:+.4f}</span>
        <span class="weight-badge">w₂ = {w2:+.4f}</span>
        <span class="weight-badge">b &nbsp;= {bias:+.4f}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="text-align:center; margin-top:0.8rem; color:#7c7ca8; font-size:0.85rem;">
        <code style="color:#a78bfa;">
            y = σ({w1:+.4f}·x₁ {'+' if w2 >= 0 else ''}{w2:.4f}·x₂ {'+' if bias >= 0 else ''}{bias:.4f})
        </code>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Row 3: Charts ────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

# ── 3D Decision Boundary Surface ─────────────────────────────────────────
with chart_col1:
    section_header("🌐", "Decision Boundary (3D)")

    resolution = 60
    xs = np.linspace(-0.3, 1.3, resolution)
    ys = np.linspace(-0.3, 1.3, resolution)
    xx, yy = np.meshgrid(xs, ys)

    with torch.no_grad():
        grid = torch.tensor(
            np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32
        )
        zz = model(grid).numpy().reshape(xx.shape)

    # Surface
    surface = go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale=[
            [0.0, "#1e1b4b"],
            [0.3, "#4c1d95"],
            [0.5, "#7c3aed"],
            [0.7, "#a78bfa"],
            [1.0, "#34d399"],
        ],
        opacity=0.88,
        showscale=True,
        colorbar=dict(
            title=dict(text="P(y=1)", font=dict(color="#c4b5fd", size=12)),
            tickfont=dict(color="#9090c0", size=10),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            len=0.6,
        ),
        name="σ surface",
    )

    # Decision boundary plane at z = 0.5
    boundary = go.Surface(
        x=xx,
        y=yy,
        z=np.full_like(xx, 0.5),
        colorscale=[[0, "rgba(251,113,133,0.25)"], [1, "rgba(251,113,133,0.25)"]],
        showscale=False,
        opacity=0.3,
        name="z = 0.5",
    )

    # OR-gate data points
    points_x = [0, 0, 1, 1]
    points_y = [0, 1, 0, 1]
    points_z = [0, 1, 1, 1]
    point_colors = ["#fb7185" if z == 0 else "#34d399" for z in points_z]

    scatter = go.Scatter3d(
        x=points_x,
        y=points_y,
        z=[p + 0.02 for p in points_z],  # slight lift so they sit above the surface
        mode="markers+text",
        marker=dict(size=7, color=point_colors, line=dict(width=1, color="#fff")),
        text=["0", "1", "1", "1"],
        textposition="top center",
        textfont=dict(color="#e0e0f0", size=11),
        name="OR truth",
    )

    # Current input marker
    current_scatter = go.Scatter3d(
        x=[x1],
        y=[x2],
        z=[prob + 0.03],
        mode="markers",
        marker=dict(
            size=10,
            color="#fbbf24",
            symbol="diamond",
            line=dict(width=2, color="#fff"),
        ),
        name=f"Current ({x1},{x2})",
    )

    fig3d = go.Figure(data=[surface, boundary, scatter, current_scatter])
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="X₁", font=dict(color="#c4b5fd")),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(120,100,255,0.12)",
                zerolinecolor="rgba(120,100,255,0.2)",
                tickfont=dict(color="#9090c0"),
            ),
            yaxis=dict(
                title=dict(text="X₂", font=dict(color="#c4b5fd")),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(120,100,255,0.12)",
                zerolinecolor="rgba(120,100,255,0.2)",
                tickfont=dict(color="#9090c0"),
            ),
            zaxis=dict(
                title=dict(text="P(y=1)", font=dict(color="#c4b5fd")),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(120,100,255,0.12)",
                zerolinecolor="rgba(120,100,255,0.2)",
                range=[0, 1.05],
                tickfont=dict(color="#9090c0"),
            ),
            bgcolor="rgba(10, 10, 30, 0.0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=480,
        legend=dict(font=dict(color="#c4b5fd", size=11), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ── Training Loss Convergence ────────────────────────────────────────────
with chart_col2:
    section_header("📉", "Training Loss Convergence")

    if history["epochs"]:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history["loss"],
                mode="lines",
                line=dict(
                    color="#a78bfa",
                    width=2.5,
                    shape="spline",
                    smoothing=1.1,
                ),
                fill="tozeroy",
                fillcolor="rgba(124, 58, 237, 0.10)",
                name="BCE Loss",
            )
        )
        fig_loss.update_layout(
            xaxis=dict(
                title="Epoch",
                color="#9090c0",
                gridcolor="rgba(120,100,255,0.08)",
                zerolinecolor="rgba(120,100,255,0.15)",
            ),
            yaxis=dict(
                title="Loss",
                color="#9090c0",
                gridcolor="rgba(120,100,255,0.08)",
                zerolinecolor="rgba(120,100,255,0.15)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=480,
            legend=dict(
                font=dict(color="#c4b5fd", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.warning("No training history found. Run `python train.py` first.")

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; margin-top:3rem; padding:1.5rem 0;
                border-top: 1px solid rgba(120,100,255,0.1);">
        <p style="color:#5c5c8a; font-size:0.78rem; letter-spacing:1px;">
            BINARY PATTERN CLASSIFIER &nbsp;·&nbsp; Single-Layer Perceptron
            &nbsp;·&nbsp; PyTorch + Streamlit + Plotly
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
