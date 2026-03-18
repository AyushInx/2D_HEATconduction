import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st
import io

st.set_page_config(page_title="Heat Conduction Solver", page_icon=None, layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
}

.stApp {
    background-color: #F7F5F2;
    color: #1a1a1a;
}

/* Top header bar */
.app-header {
    border-bottom: 1px solid #d0ccc6;
    padding-bottom: 18px;
    margin-bottom: 32px;
}

.app-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: #1a1a1a;
    letter-spacing: 0.04em;
    margin: 0;
    line-height: 1.2;
}

.app-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #888;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.0rem;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: 0.03em;
    border-bottom: 1px solid #e0ddd8;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

/* Metric cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #e8e4de;
    border-radius: 4px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.metric-card .label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #1a1a1a;
    line-height: 1;
}

/* Run button */
div[data-testid="stButton"] > button {
    background-color: #1a1a1a;
    color: #F7F5F2;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 14px 28px;
    width: 100%;
    transition: background 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background-color: #333;
}

/* Download buttons — white with border */
div[data-testid="stDownloadButton"] > button {
    background-color: #FFFFFF !important;
    color: #1a1a1a !important;
    border: 1px solid #c0bdb8 !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 12px 20px !important;
    width: 100% !important;
    transition: background 0.2s, border-color 0.2s !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background-color: #f0ede8 !important;
    border-color: #1a1a1a !important;
}

/* Fix number input +/- buttons */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: #f0ede8 !important;
    color: #1a1a1a !important;
    border: 1px solid #ddd !important;
    width: auto !important;
    padding: 4px 10px !important;
    font-size: 1rem !important;
}
button[data-testid="stNumberInputStepUp"]:hover,
button[data-testid="stNumberInputStepDown"]:hover {
    background-color: #e0ddd8 !important;
}

/* Fix checkboxes */
input[type="checkbox"] {
    accent-color: #1a1a1a !important;
}

/* Input labels */
.stNumberInput label, .stSlider label,
.stSelectbox label, .stCheckbox label,
.stSelectSlider label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #555 !important;
    letter-spacing: 0.02em !important;
}

/* Input boxes */
input[type="number"] {
    background: #FFFFFF !important;
    border: 1px solid #ddd !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    color: #1a1a1a !important;
}

/* Divider */
hr { border-color: #e0ddd8 !important; }

/* Info box */
.stAlert {
    background: #FFFFFF !important;
    border: 1px solid #e0ddd8 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #555 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
}
.stTabs [aria-selected="true"] {
    color: #1a1a1a !important;
    border-bottom: 2px solid #1a1a1a !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Core functions ────────────────────────────────────────────────────
def create_mesh(nx, ny, tTop=0, tBottom=0, tLeft=0, tRight=0):
    mesh = np.zeros((ny, nx), dtype=float)
    mesh[0, :] = tBottom
    mesh[ny-1, :] = tTop
    mesh[:, 0] = tLeft
    mesh[:, nx-1] = tRight
    return mesh


def iterate(tMesh, nx, ny, tolerance, h=0, k=1, Tinf=0,
            dirichlet_top=True, dirichlet_bottom=True,
            dirichlet_left=True, dirichlet_right=True):
    iterations = 0
    maxIterations = 100_000
    error = 100.0
    Bi = h / k

    progress = st.progress(0, text="Computing solution...")

    while error > tolerance and iterations < maxIterations:
        error = 0.0
        iterations += 1
        tOld = tMesh.copy()

        for j in range(1, nx-1):
            for i in range(1, ny-1):
                tMesh[i, j] = 0.25 * (tMesh[i+1,j] + tMesh[i-1,j] + tMesh[i,j+1] + tMesh[i,j-1])
                error = max(error, abs(tMesh[i,j] - tOld[i,j]))

        if not dirichlet_bottom:
            for j in range(1, nx-1):
                tMesh[0,j] = (tMesh[0,j+1] + tMesh[0,j-1] + tMesh[1,j] + Bi*Tinf) / (3+Bi)
        if not dirichlet_top:
            for j in range(1, nx-1):
                tMesh[ny-1,j] = (tMesh[ny-1,j+1] + tMesh[ny-1,j-1] + tMesh[ny-2,j] + Bi*Tinf) / (3+Bi)
        if not dirichlet_left:
            for i in range(1, ny-1):
                tMesh[i,0] = (tMesh[i+1,0] + tMesh[i-1,0] + tMesh[i,1] + Bi*Tinf) / (3+Bi)
        if not dirichlet_right:
            for i in range(1, ny-1):
                tMesh[i,nx-1] = (tMesh[i+1,nx-1] + tMesh[i-1,nx-1] + tMesh[i,nx-2] + Bi*Tinf) / (3+Bi)

        if not dirichlet_bottom and not dirichlet_right:
            tMesh[0,nx-1] = (tMesh[0,nx-2] + tMesh[1,nx-1] + 2*Bi*Tinf) / (2+2*Bi)
        if not dirichlet_bottom and not dirichlet_left:
            tMesh[0,0] = (tMesh[0,1] + tMesh[1,0] + 2*Bi*Tinf) / (2+2*Bi)
        if not dirichlet_top and not dirichlet_right:
            tMesh[ny-1,nx-1] = (tMesh[ny-1,nx-2] + tMesh[ny-2,nx-1] + 2*Bi*Tinf) / (2+2*Bi)
        if not dirichlet_top and not dirichlet_left:
            tMesh[ny-1,0] = (tMesh[ny-1,1] + tMesh[ny-2,0] + 2*Bi*Tinf) / (2+2*Bi)

        if iterations % 200 == 0:
            progress.progress(min(iterations/maxIterations, 1.0),
                              text=f"Iteration {iterations}  —  residual {error:.2e}")

    progress.progress(1.0, text=f"Converged  |  {iterations} iterations  |  residual {error:.2e}")
    return tMesh, iterations, error


def theoretical_solution(x, y, Lx, Ly, tTop, tBottom, tLeft, tRight, terms=100):
    tT = tL = tR = 0.0
    for n in range(1, terms, 2):
        tT += (4*(tTop-tBottom)/(n*np.pi)) * np.sinh(n*np.pi*y/Lx) / np.sinh(n*np.pi*Ly/Lx) * np.sin(n*np.pi*x/Lx)
        tL += (4*(tLeft-tBottom)/(n*np.pi)) * np.sinh(n*np.pi*x/Ly) / np.sinh(n*np.pi*Lx/Ly) * np.sin(n*np.pi*y/Ly)
        tR += (4*(tRight-tBottom)/(n*np.pi)) * np.sinh(n*np.pi*(Lx-x)/Ly) / np.sinh(n*np.pi*Lx/Ly) * np.sin(n*np.pi*y/Ly)
    return tT + tBottom + tL + tR


def analytical_grid(nx, ny, Lx, Ly, tTop, tBottom, tLeft, tRight):
    grid = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            grid[i,j] = theoretical_solution(j/(nx-1)*Lx, i/(ny-1)*Ly, Lx, Ly, tTop, tBottom, tLeft, tRight)
    return grid


def make_fig(mesh, title, cmap="magma", isotherms=True):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#F7F5F2")
    ax.set_facecolor("#F7F5F2")
    cf = ax.contourf(mesh, 200, cmap=cmap)
    cb = fig.colorbar(cf, ax=ax)
    cb.ax.yaxis.set_tick_params(color="#666")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#555", fontsize=8,
             fontfamily="monospace")
    cb.set_label("Temperature (°C)", color="#555", fontsize=9,
                 fontfamily="monospace")
    if isotherms:
        cs = ax.contour(mesh, 20, colors="white", alpha=0.4, linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=7, colors="white", fmt="%.0f")
    ax.set_title(title, color="#1a1a1a", fontfamily="serif",
                 fontsize=11, pad=12, fontweight="light")
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ddd")
    fig.tight_layout()
    return fig


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">Heat Conduction Solver</div>
  <div class="app-subtitle">2D Steady-State &nbsp;·&nbsp; Finite Difference &nbsp;·&nbsp; Gauss-Seidel Iteration</div>
</div>
""", unsafe_allow_html=True)

# ── Domain ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Domain</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
Lx = d1.number_input("Length  Lx", value=1.0, min_value=0.1, step=0.1)
Ly = d2.number_input("Height  Ly", value=1.0, min_value=0.1, step=0.1)
nx = d3.slider("Grid columns  (nx)", 5, 100, 40)
ny = d4.slider("Grid rows  (ny)", 5, 100, 40)

# ── Boundary Temperatures ─────────────────────────────────────────────
st.markdown('<div class="section-label">Boundary Temperatures — °C</div>', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
tTop    = b1.number_input("Top",    value=100.0)
tBottom = b2.number_input("Bottom", value=0.0)
tLeft   = b3.number_input("Left",   value=0.0)
tRight  = b4.number_input("Right",  value=0.0)

# ── Solver ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Solver</div>', unsafe_allow_html=True)
tolerance = st.select_slider(
    "Convergence Tolerance",
    options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    value=1e-6,
    format_func=lambda x: f"{x:.0e}"
)

# ── Convection ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Convection — Optional</div>', unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="stCheckbox"] label p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
}
div[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #1a1a1a !important;
    width: 16px !important;
    height: 16px !important;
}
</style>
""", unsafe_allow_html=True)
use_conv = st.checkbox("Enable convective boundary conditions")
if use_conv:
    cv1, cv2, cv3 = st.columns(3)
    h    = cv1.number_input("h  (W/m²K)", value=10.0, min_value=0.0)
    k    = cv2.number_input("k  (W/mK)",  value=1.0,  min_value=0.01)
    Tinf = cv3.number_input("T-infinity  (°C)", value=25.0)
    st.caption("Select faces with convection — all others retain fixed temperature")

    # Colored face selector cards
    face_colors = {
        "Top":    {"bg": "#4A90D9", "hover": "#3578C4"},
        "Bottom": {"bg": "#E8704A", "hover": "#D45D37"},
        "Left":   {"bg": "#5BAD6F", "hover": "#47995B"},
        "Right":  {"bg": "#9B6BB5", "hover": "#855EA0"},
    }

    for face in ["conv_top", "conv_bottom", "conv_left", "conv_right"]:
        if face not in st.session_state:
            st.session_state[face] = False

    st.markdown("""
    <style>
    div[data-testid="stButton"].face-btn > button {
        width: 100% !important;
        padding: 18px 10px !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        transition: opacity 0.2s !important;
    }
    </style>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)

    faces = [
        ("conv_top",    "Top",    "#4A90D9", f1),
        ("conv_bottom", "Bottom", "#E8704A", f2),
        ("conv_left",   "Left",   "#5BAD6F", f3),
        ("conv_right",  "Right",  "#9B6BB5", f4),
    ]

    for key, label, color, col in faces:
        active = st.session_state[key]
        bg     = color if active else "#FFFFFF"
        fg     = "#FFFFFF" if active else "#555"
        border = color
        tag    = "ON" if active else "OFF"
        with col:
            st.markdown(
                f"""<div style="background:{bg}; color:{fg}; border:2px solid {border};
                border-radius:6px; padding:14px 10px; text-align:center;
                font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                font-weight:600; letter-spacing:0.08em; cursor:pointer;
                margin-bottom:4px;">
                {label}<br><span style="font-size:0.6rem; opacity:0.85;">{tag}</span>
                </div>""",
                unsafe_allow_html=True
            )
            if st.button(f"Toggle {label}", key=f"btn_{key}"):
                st.session_state[key] = not st.session_state[key]
                st.rerun()

    conv_top    = st.session_state["conv_top"]
    conv_bottom = st.session_state["conv_bottom"]
    conv_left   = st.session_state["conv_left"]
    conv_right  = st.session_state["conv_right"]

    dirichlet_top    = not conv_top
    dirichlet_bottom = not conv_bottom
    dirichlet_left   = not conv_left
    dirichlet_right  = not conv_right
else:
    h = 0; k = 1; Tinf = 0
    dirichlet_top = dirichlet_bottom = dirichlet_left = dirichlet_right = True

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("Run Solver")

if not run:
    st.info("Configure the parameters above and click Run Solver to compute the temperature field.")
    st.stop()

# ── Solve ─────────────────────────────────────────────────────────────
mesh = create_mesh(nx, ny, tTop, tBottom, tLeft, tRight)
mesh, iters, final_err = iterate(mesh, nx, ny, tolerance, h, k, Tinf,
                                  dirichlet_top, dirichlet_bottom,
                                  dirichlet_left, dirichlet_right)

# ── Metrics ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "Iterations",      f"{iters:,}"),
    (c2, "Residual",        f"{final_err:.2e}"),
    (c3, "Min Temp  (°C)",  f"{mesh.min():.2f}"),
    (c4, "Max Temp  (°C)",  f"{mesh.max():.2f}"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Plots ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Temperature Field</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Distribution", "Isotherms", "Combined"])

with tab1:
    st.pyplot(make_fig(mesh, "Temperature Distribution", cmap="magma", isotherms=False))

with tab2:
    fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor="#F7F5F2")
    ax2.set_facecolor("#F7F5F2")
    cs = ax2.contour(mesh, 20, cmap="coolwarm")
    ax2.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    ax2.set_title("Isotherms", color="#1a1a1a", fontfamily="serif",
                  fontsize=11, pad=12, fontweight="light")
    ax2.tick_params(colors="#888", labelsize=8)
    for s in ax2.spines.values(): s.set_edgecolor("#ddd")
    fig2.tight_layout()
    st.pyplot(fig2)

with tab3:
    st.pyplot(make_fig(mesh, "Temperature Distribution with Isotherms", cmap="magma", isotherms=True))

# ── Analytical comparison ─────────────────────────────────────────────
if not use_conv:
    st.markdown('<div class="section-label">Analytical Validation</div>', unsafe_allow_html=True)
    with st.spinner("Computing analytical solution..."):
        theory = analytical_grid(nx, ny, Lx, Ly, tTop, tBottom, tLeft, tRight)
    err_field = np.abs(mesh - theory)

    ac1, ac2 = st.columns(2)
    with ac1:
        st.pyplot(make_fig(theory, "Analytical Solution", cmap="magma", isotherms=False))
    with ac2:
        st.pyplot(make_fig(err_field, "Absolute Error  |Numerical − Analytical|",
                           cmap="YlOrRd", isotherms=False))

    e1, e2 = st.columns(2)
    e1.markdown(
        f'<div class="metric-card"><div class="label">Max Error</div>'
        f'<div class="value">{err_field.max():.4f}</div></div>',
        unsafe_allow_html=True
    )
    e2.markdown(
        f'<div class="metric-card"><div class="label">Mean Error</div>'
        f'<div class="value">{err_field.mean():.4f}</div></div>',
        unsafe_allow_html=True
    )

# ── Downloads ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)

csv_buf = io.StringIO()
np.savetxt(csv_buf, mesh, delimiter=",", fmt="%.6f")
dl1.download_button("Download  —  Temperature Numerical (.csv)",
                    csv_buf.getvalue(), "Temperature_Numerical.csv", "text/csv")

txt_lines = ["  ".join(f"{v:10.4f}" for v in row) for row in mesh]
dl2.download_button("Download  —  Temperature Numerical (.txt)",
                    "\n".join(txt_lines), "Temperature_Numerical.txt", "text/plain")
