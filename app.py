
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Atmos | Weather Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #0b1120;
    color: #e5edf9;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f0f6ff !important;
}

p, label, span {
    color: #d2dced;
}

[data-testid="stSidebar"] {
    background: #111a2c;
    border-right: 1px solid #25324a;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243a, #111a2c);
    padding: 20px;
    border: 1px solid #293a56;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] {
    color: #9eafca !important;
}

[data-testid="stMetricValue"] {
    color: #f1f7ff !important;
    font-family: 'Space Grotesk', sans-serif;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(115deg, #172b4b 0%, #173b55 52%, #25316a 100%);
    border: 1px solid #344b70;
    margin-bottom: 25px;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 11px;
    color: #73e5db;
    font-weight: 700;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    margin-top: 12px;
}

.hero-subtitle {
    font-size: 15px;
    color: #c0d0e8;
    max-width: 700px;
    line-height: 1.8;
    margin-top: 12px;
}

.section-label {
    color: #70e2d8;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-card {
    padding: 24px;
    background: linear-gradient(145deg, #17243a, #101a2c);
    border: 1px solid #2c3d58;
    border-radius: 18px;
    margin: 12px 0 20px 0;
}

.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 54px;
    line-height: 1.2;
    font-weight: 700;
    color: #74e6d9;
}

.score-caption {
    color: #9eafca;
    font-size: 13px;
}

div[data-testid="stTabs"] button {
    color: #bac9df;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #75e4da;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    border: 1px solid #4a7b91;
    background: #173d50;
    color: white;
    font-weight: 600;
}

.stButton button:hover, .stDownloadButton button:hover {
    border-color: #70e2d8;
    color: #70e2d8;
}

hr {
    border-color: #26364d;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD THE MODEL FROM weather.ipynb
# =====================================================
import ast
import json
from pathlib import Path

@st.cache_resource
def load_notebook_model(notebook_path):
    """Load the model definitions from weather.ipynb (without running plots/tests)."""
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Could not find {notebook_path.name}. Put weather.ipynb in the same folder as app.py."
        )

    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    namespace = {"np": np, "pd": pd, "plt": plt, "sns": sns, "fuzz": fuzz, "ctrl": ctrl}
    cells = notebook.get("cells", [])

    # These notebook cells define the fuzzy variables, membership functions,
    # 27 rules, and the Mamdani control system. Plotting/experiment cells are skipped.
    for cell_index in (1, 2, 4, 5):
        if cell_index >= len(cells):
            raise ValueError(f"weather.ipynb is missing required cell {cell_index}.")
        source = "".join(cells[cell_index].get("source", []))
        exec(compile(source, f"weather.ipynb cell {cell_index}", "exec"), namespace)

    # Reuse the exact predict_comfort and baseline_comfort function definitions
    # from the notebook, without running its example predictions or experiments.
    for cell_index in (6, 9):
        source = "".join(cells[cell_index].get("source", []))
        tree = ast.parse(source)
        function_nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not function_nodes:
            raise ValueError(f"Could not find a function definition in notebook cell {cell_index}.")
        module = ast.Module(body=function_nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), f"weather.ipynb cell {cell_index}", "exec"), namespace)

    return namespace


notebook_dir = Path(__file__).resolve().parent
notebook_path = notebook_dir / "weather.ipynb"
if not notebook_path.exists():
    # Also support the original uploaded filename.
    notebook_path = notebook_dir / "weather(2).ipynb"
try:
    notebook_model = load_notebook_model(str(notebook_path))
except Exception as exc:
    st.error(f"Unable to load the fuzzy model from weather.ipynb: {exc}")
    st.stop()

temperature = notebook_model["temperature"]
humidity = notebook_model["humidity"]
wind_speed = notebook_model["wind_speed"]
comfort = notebook_model["comfort"]
rules = notebook_model["rules"]
system = notebook_model["comfort_system"]
predict_comfort = notebook_model["predict_comfort"]
baseline_comfort = notebook_model["baseline_comfort"]


def get_category(score):
    if score <= 25:
        return "Poor"
    elif score <= 50:
        return "Moderate"
    elif score <= 75:
        return "Comfortable"
    return "Excellent"


# =====================================================
# SIDEBAR — USER INPUTS
# =====================================================
with st.sidebar:
    st.markdown("## 🌤️ ATMOS")
    st.caption("WEATHER INTELLIGENCE")
    st.divider()

    st.markdown("### Configure conditions")
    st.write("Adjust the weather parameters to calculate a fuzzy comfort score.")

    t = st.slider(
        "🌡️ Temperature (°C)",
        min_value=0, max_value=45, value=28
    )
    h = st.slider(
        "💧 Relative humidity (%)",
        min_value=0, max_value=100, value=70
    )
    w = st.slider(
        "💨 Wind speed (km/h)",
        min_value=0, max_value=50, value=12
    )

    st.divider()
    st.markdown("#### Inference engine")
    st.markdown("**Mamdani FIS**")
    st.caption("MIN implication · MAX aggregation · Centroid defuzzification")
    st.caption("27 illustrative IF–THEN rules")


# =====================================================
# PREDICTION
# =====================================================
score = predict_comfort(t, h, w)
base_score = baseline_comfort(t, h, w)
category = get_category(score)


# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI & SOFT COMPUTING PROJECT</div>
    <div class="hero-title">Weather Comfort<br>Intelligence.</div>
    <div class="hero-subtitle">
        An interactive weather comfort estimator powered by a
        Mamdani Fuzzy Inference System. Adjust environmental inputs,
        inspect the inference results, and explore how the model behaves.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# TOP METRICS
# =====================================================
st.markdown('<div class="section-label">CURRENT INPUTS</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Temperature", f"{t} °C")
c2.metric("Humidity", f"{h}%")
c3.metric("Wind speed", f"{w} km/h")

st.markdown("")


# =====================================================
# COMFORT SCORE CARD
# =====================================================
left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="section-label">FUZZY PREDICTION</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="score-caption">COMFORT SCORE</div>
        <div class="score-number">{score:.1f}<span style="font-size:22px;color:#9eafca"> / 100</span></div>
        <div style="font-size:18px;font-weight:600;color:#f0f6ff;margin-top:8px">
            {category}
        </div>
        <div class="score-caption" style="margin-top:8px">
            Centroid-defuzzified Mamdani output
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(np.clip(round(score), 0, 100)))
    st.caption("The score is generated by the configured fuzzy rules.")

with right:
    st.markdown('<div class="section-label">MODEL COMPARISON</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="score-caption">ILLUSTRATIVE BASELINE</div>
        <div class="score-number" style="color:#a9baff">
            {base_score:.1f}
        </div>
        <div class="score-caption" style="margin-top:12px">
            Difference from Mamdani:
            <b>{abs(score - base_score):.1f} points</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "The baseline is a hand-crafted comparison formula, "
        "not a measured ground-truth model."
    )


# =====================================================
# VISUALIZATION TABS
# =====================================================
st.markdown("")
st.markdown('<div class="section-label">EXPLORE THE MODEL</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Membership functions",
    "📊 Score comparison",
    "🗺️ Comfort heatmap",
    "🧠 Rule base"
])


# Membership functions
with tab1:
    st.subheader("Fuzzification")
    st.write(
        "Each input is mapped to linguistic fuzzy sets with membership "
        "degrees between 0 and 1."
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#111a2c")

    variables = [
        (temperature, "Temperature (°C)"),
        (humidity, "Humidity (%)"),
        (wind_speed, "Wind speed (km/h)"),
        (comfort, "Comfort score")
    ]

    for ax, (variable, title) in zip(axes.flat, variables):
        ax.set_facecolor("#111a2c")

        for term in variable.terms:
            ax.plot(
                variable.universe,
                variable[term].mf,
                label=term.capitalize(),
                linewidth=2
            )

        ax.set_title(title, color="white")
        ax.set_xlabel("Value", color="white")
        ax.set_ylabel("Membership degree", color="white")
        ax.tick_params(colors="#b8c7dc")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.15)
        ax.legend(facecolor="#17243a", labelcolor="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Score comparison
with tab2:
    st.subheader("Mamdani vs baseline")
    st.write(
        "The baseline is illustrative. The graph compares behavior, "
        "not prediction accuracy against real-world labels."
    )

    temp_values = np.arange(0, 46, 1)
    fuzzy_values = [
        predict_comfort(x, h, w) for x in temp_values
    ]
    base_values = [
        baseline_comfort(x, h, w) for x in temp_values
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(temp_values, fuzzy_values, label="Mamdani FIS", linewidth=2.5)
    ax.plot(
        temp_values, base_values,
        "--", label="Illustrative baseline", linewidth=2
    )
    ax.axvline(t, linestyle=":", label="Selected temperature")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Comfort score")
    ax.set_ylim(0, 100)
    ax.set_title(f"Sensitivity at humidity={h}% and wind={w} km/h")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


# Heatmap
with tab3:
    st.subheader("Temperature and humidity")
    st.write(
        "Explore predicted scores over a grid of temperature and humidity "
        "values. Wind speed stays fixed at the selected input."
    )

    temp_values = np.arange(0, 46, 3)
    humidity_values = np.arange(0, 101, 5)

    values = np.array([
        [predict_comfort(x, y, w) for x in temp_values]
        for y in humidity_values
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        values,
        xticklabels=temp_values,
        yticklabels=humidity_values,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "Comfort score"}
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title(f"Comfort landscape | Wind={w} km/h")
    st.pyplot(fig)
    plt.close(fig)


# Rule base
with tab4:
    st.subheader("Mamdani IF–THEN rules")
    st.write(
        "The system uses 27 combinations of low, medium, and high "
        "input categories."
    )

    temp_score_map = {"low": 1, "medium": 3, "high": 1}
    hum_score_map = {"low": 0, "medium": 0, "high": -1}
    wind_score_map = {"low": 1, "medium": 0, "high": -1}

    rows = []

    for tl in ["low", "medium", "high"]:
        for hl in ["low", "medium", "high"]:
            for wl in ["low", "medium", "high"]:
                value = (
                    temp_score_map[tl]
                    + hum_score_map[hl]
                    + wind_score_map[wl]
                )

                if value <= 0:
                    output = "Poor"
                elif value == 1:
                    output = "Moderate"
                elif value == 2:
                    output = "Comfortable"
                else:
                    output = "Excellent"

                rows.append({
                    "Temperature": tl,
                    "Humidity": hl,
                    "Wind speed": wl,
                    "Comfort output": output
                })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# DOWNLOAD RESULT
# =====================================================
st.divider()
st.subheader("Export prediction")

result = pd.DataFrame([{
    "Temperature_C": t,
    "Humidity_Percent": h,
    "Wind_Speed_kmh": w,
    "Mamdani_Score": round(score, 2),
    "Comfort_Category": category,
    "Baseline_Score": round(base_score, 2)
}])

st.download_button(
    "⬇️ Download result as CSV",
    data=result.to_csv(index=False),
    file_name="weather_comfort_prediction.csv",
    mime="text/csv"
)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#8193ad;font-size:12px'>"
    "ATMOS · Mamdani Fuzzy Inference System · Educational project"
    "</div>",
    unsafe_allow_html=True
)
