from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_loader import PreferenceFileError, load_model_from_excel
from election_engine import (
    SimulationParameters,
    run_monte_carlo,
    run_single_simulation,
)

GA_MEASUREMENT_ID = "G-Y5NQPPCSPS"
MAX_VOTERS = 39

PRIMARY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
PROBABILITY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def inject_google_analytics(measurement_id: str) -> None:
    ga_code = f"""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{measurement_id}', {{
        'debug_mode': true,
        'send_page_view': true
      }});
    </script>
    """

    if hasattr(st, "html"):
        st.html(ga_code, unsafe_allow_javascript=True)
    else:
        components.html(ga_code, height=0, width=0)


st.set_page_config(page_title="საპატრიარქო არჩევნების სიმულაცია", page_icon="🗳️", layout="wide")
inject_google_analytics(GA_MEASUREMENT_ID)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 14px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p,
    div[data-testid="stMetric"] span,
    div[data-testid="stMetric"] div:not([data-testid="stMetricValue"]),
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #0f172a !important;
    }

    .winner-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.18);
    }

    .winner-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #166534;
        margin-bottom: 6px;
    }

    .winner-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #14532d;
        line-height: 1.15;
    }

    .winner-subtext {
        margin-top: 8px;
        font-size: 0.95rem;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = Path(__file__).parent / "data" / "preferences.xlsx"


@st.cache_data(show_spinner=False)
def load_default_model(path_str: str):
    return load_model_from_excel(path_str)


def build_preference_matrix(model) -> pd.DataFrame:
    rows = []
    for elector in model.electors:
        row = {"მღვდელმთავარი": elector}
        row.update(model.preferences[elector])
        rows.append(row)
    return pd.DataFrame(rows)


def totals_to_df(vote_totals: dict[str, int], value_label: str = "ხმები") -> pd.DataFrame:
    return pd.DataFrame({"კანდიდატი": list(vote_totals.keys()), value_label: list(vote_totals.values())})


def _style_axes(ax, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=18, labelsize=10)
    ax.tick_params(axis="y", labelsize=9)


def render_vote_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    y_max: float | None = None,
    bar_width: float = 0.56,
    figsize: tuple[float, float] = (7.0, 4.15),
):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    colors = PRIMARY_COLORS[: len(df)]
    bars = ax.bar(df[category_col], df[value_col], color=colors, width=bar_width)

    _style_axes(ax, title, category_col, value_col)

    upper = y_max if y_max is not None else max(float(df[value_col].max()) + 1, 1)
    ax.set_ylim(0, upper)
    ax.margins(x=0.08)

    label_offset = max(upper * 0.012, 0.12)
    for bar, value in zip(bars, df[value_col]):
        text_y = value + label_offset
        va = "bottom"
        if text_y >= upper:
            text_y = value - label_offset
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{value}",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_probability_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    bar_width: float = 0.54,
    figsize: tuple[float, float] = (7.0, 4.0),
):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    colors = PROBABILITY_COLORS[: len(df)]
    bars = ax.bar(df[category_col], df[value_col], color=colors, width=bar_width)

    _style_axes(ax, title, category_col, value_col)

    upper = max(float(df[value_col].max()) * 1.15, 1.0)
    ax.set_ylim(0, upper)
    ax.margins(x=0.08)

    label_offset = max(upper * 0.012, 0.18)
    for bar, value in zip(bars, df[value_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.title("საარჩევნო სიმულატორი")
st.caption("აპლიკაცია მონაცემებს ავტომატურად კითხულობს repo-ში შენახული Excel ფაილიდან და თითო არჩევნებში იყენებს ზუსტად 3 კანდიდატს.")

if not DATA_FILE.exists():
    st.error(
        "მონაცემთა ფაილი ვერ მოიძებნა. repo-ში უნდა არსებობდეს ფაილი ამ ბილიკზე: "
        f"{DATA_FILE.as_posix()}"
    )
    st.stop()

try:
    model = load_default_model(str(DATA_FILE))
except PreferenceFileError as exc:
    st.error(f"ფაილის სტრუქტურის შეცდომა: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"ფაილის წაკითხვა ვერ მოხერხდა: {exc}")
    st.stop()

with st.sidebar:
    st.header("პარამეტრები")
    st.caption(f"მონაცემთა ფაილი: {DATA_FILE.name}")

    default_candidates = model.candidates[:3]

    with st.form("simulation_form"):
        selected_candidates = st.multiselect(
            "აირჩიე 3 კანდიდატი",
            options=model.candidates,
            default=default_candidates,
            max_selections=3,
        )

        iterations = st.number_input("იტერაციების რაოდენობა", min_value=1, max_value=200_000, value=100, step=500)
        volatility_level = st.slider("რყევადობის დონე", min_value=0, max_value=10, value=3)
        elector_absence_pct = st.slider("ამომრჩევლის გაცდენის შანსი (%)", min_value=0, max_value=100, value=5)
        candidate_absence_pct = st.slider("კანდიდატის გაცდენის შანსი (%)", min_value=0, max_value=100, value=0)
        rng_seed_text = st.text_input("Seed (სურვილისამებრ, გამეორებადი შედეგებისთვის)", value="")

        submitted = st.form_submit_button("სიმულაციის გაშვება", use_container_width=True)

if "simulation_payload" not in st.session_state:
    st.session_state.simulation_payload = None

if submitted:
    if len(selected_candidates) != 3:
        st.error("სიმულაციის გასაშვებად ზუსტად 3 კანდიდატი უნდა აირჩიო.")
        st.stop()

    try:
        rng_seed = int(rng_seed_text) if rng_seed_text.strip() else None
    except ValueError:
        st.error("Seed უნდა იყოს მთელი რიცხვი.")
        st.stop()

    st.session_state.simulation_payload = {
        "selected_candidates": selected_candidates,
        "iterations": int(iterations),
        "volatility_level": int(volatility_level),
        "elector_absence_pct": int(elector_absence_pct),
        "candidate_absence_pct": int(candidate_absence_pct),
        "rng_seed": rng_seed,
    }

payload = st.session_state.simulation_payload
if payload is None:
    st.info("აირჩიე პარამეტრები და დააჭირე „სიმულაციის გაშვება“.")
    st.stop()

params = SimulationParameters(
    selected_candidates=payload["selected_candidates"],
    volatility_level=payload["volatility_level"],
    elector_absence_probability=payload["elector_absence_pct"] / 100,
    candidate_absence_probability=payload["candidate_absence_pct"] / 100,
)

try:
    single_result = run_single_simulation(model, params)
    monte_carlo_result = run_monte_carlo(model, params, payload["iterations"], rng_seed=payload["rng_seed"])
except Exception as exc:
    st.error(f"სიმულაციის გაშვებისას დაფიქსირდა შეცდომა: {exc}")
    st.stop()

st.subheader("ერთი კონკრეტული გაშვების დეტალები")

winner_text = single_result["winner"]
winner_note = (
    "პირველივე ტურში დაფიქსირდა გამარჯვება."
    if not single_result["runoff_required"]
    else "გამარჯვებული გამოვლინდა მეორე ტურის შემდეგ."
)

st.markdown(
    f"""
    <div class="winner-card">
        <div class="winner-label">საბოლოო შედეგი</div>
        <div class="winner-name">{winner_text}</div>
        <div class="winner-subtext">{winner_note}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
col1.metric("დასწრება", single_result["attendance_count"])
col2.metric("გაცდენა", single_result["absence_count"])

if single_result["absent_candidates"]:
    st.write("**გაცდენილი კანდიდატები:**", ", ".join(single_result["absent_candidates"]))
else:
    st.write("**გაცდენილი კანდიდატები:** არავინ")

first_round = single_result["first_round"]
st.markdown("### პირველი ტური")
fr_col1, fr_col2, fr_col3 = st.columns(3)
fr_col1.metric("ვალიდური ხმები", first_round["valid_votes"])
fr_col2.metric("გამარჯვების ბარიერი", first_round["majority_threshold"])
fr_col3.metric(
    "სტატუსი",
    "მეორე ტური" if single_result["runoff_required"] else "გამარჯვებული გამოვლინდა",
)

first_vote_df = totals_to_df(first_round["vote_totals"])
left, right = st.columns([0.95, 1.05])
with left:
    st.table(first_vote_df.set_index("კანდიდატი"))
with right:
    render_vote_chart(
        first_vote_df,
        "კანდიდატი",
        "ხმები",
        "პირველი ტურის ხმები",
        y_max=MAX_VOTERS,
        bar_width=0.54,
        figsize=(5.8, 3.7),
    )

if single_result["runoff_required"]:
    st.markdown("### მეორე ტური")
    if single_result["runoff_candidates"] is None:
        st.warning("მეორე ტურის შემადგენლობა ვერ განისაზღვრა, რადგან პირველ ტურში კვალიფიკაცია ფრის გამო გაურკვეველია.")
    else:
        st.write("**მეორე ტურში გადავიდნენ:**", ", ".join(single_result["runoff_candidates"]))
        runoff = single_result["runoff"]
        runoff_df = totals_to_df(runoff["vote_totals"])
        ro_col1, ro_col2 = st.columns(2)
        ro_col1.metric("მეორე ტურის ვალიდური ხმები", runoff["valid_votes"])
        ro_col2.metric("მეორე ტურის შედეგი", single_result["winner"])

        left, right = st.columns([0.92, 1.00])
        with left:
            st.table(runoff_df.set_index("კანდიდატი"))
        with right:
            render_vote_chart(
                runoff_df,
                "კანდიდატი",
                "ხმები",
                "მეორე ტურის ხმები",
                y_max=max(float(runoff_df["ხმები"].max()) + 1, 1),
                bar_width=0.48,
                figsize=(5.2, 3.4),
            )

st.subheader("საბოლოო გამარჯვების ალბათობები")
probabilities_df = monte_carlo_result["probabilities_df"]

metrics_cols = st.columns(len(probabilities_df))
for col, (_, row) in zip(metrics_cols, probabilities_df.iterrows()):
    col.metric(row["შედეგი"], f"{row['მოგების ალბათობა (%)']:.2f}%")

left, right = st.columns([1.08, 0.92])
with left:
    st.table(probabilities_df.set_index("შედეგი"))
with right:
    render_probability_chart(
        probabilities_df,
        "შედეგი",
        "მოგების ალბათობა (%)",
        "საბოლოო გამარჯვების ალბათობები",
        bar_width=0.56,
        figsize=(7.2, 4.0),
    )

st.subheader("Monte Carlo დამატებითი მაჩვენებლები")
mc1, mc2 = st.columns(2)
mc1.metric("საშუალო დასწრება", monte_carlo_result["avg_attendance"])
mc2.metric("მეორე ტურის დანიშვნის სიხშირე", f"{monte_carlo_result['runoff_rate_pct']:.2f}%")

with st.expander("პირველი ტურის სრული ცხრილი"):
    st.dataframe(first_round["details_df"], use_container_width=True, hide_index=True)

if single_result["runoff_required"] and single_result["runoff"] is not None:
    with st.expander("მეორე ტურის სრული ცხრილი"):
        st.dataframe(single_result["runoff"]["details_df"], use_container_width=True, hide_index=True)

with st.expander("პრეფერენციების მატრიცა"):
    st.dataframe(build_preference_matrix(model), use_container_width=True, hide_index=True)
