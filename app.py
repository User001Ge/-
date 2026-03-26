from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

GA_MEASUREMENT_ID = "G-Y5NQPPCSPS"


def inject_google_analytics(measurement_id: str) -> None:
    ga_tracking_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
      </head>
      <body>
        <script>
          (function() {{
            const measurementId = {measurement_id!r};
            const initKey = `ga4_initialized_${{measurementId}}`;
            const pageKey = `ga4_last_page_${{measurementId}}`;

            window.dataLayer = window.dataLayer || [];
            function gtag() {{ dataLayer.push(arguments); }}

            function getTopLocation() {{
              try {{
                return window.parent.location.href;
              }} catch (error) {{
                return window.location.href;
              }}
            }}

            function buildPageData() {{
              const href = getTopLocation();
              try {{
                const url = new URL(href);
                return {{
                  page_location: href,
                  page_path: `${{url.pathname}}${{url.search}}`,
                  page_title: document.title || "Streamlit App"
                }};
              }} catch (error) {{
                return {{
                  page_location: href,
                  page_path: href,
                  page_title: document.title || "Streamlit App"
                }};
              }}
            }}

            const pageData = buildPageData();

            if (!sessionStorage.getItem(initKey)) {{
              gtag('js', new Date());
              gtag('config', measurementId, pageData);
              sessionStorage.setItem(initKey, 'true');
              sessionStorage.setItem(pageKey, pageData.page_path);
              return;
            }}

            const previousPage = sessionStorage.getItem(pageKey);
            if (previousPage !== pageData.page_path) {{
              gtag('event', 'page_view', pageData);
              sessionStorage.setItem(pageKey, pageData.page_path);
            }}
          }})();
        </script>
      </body>
    </html>
    """
   st.html(ga_code, unsafe_allow_javascript=True)

from data_loader import PreferenceFileError, load_model_from_excel
from election_engine import (
    SimulationParameters,
    run_monte_carlo,
    run_single_simulation,
)

st.set_page_config(page_title="არჩევნების სიმულატორი", page_icon="🗳️", layout="wide")
inject_google_analytics(GA_MEASUREMENT_ID)

DATA_FILE = Path(__file__).parent / "data" / "preferences.xlsx"
BAR_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


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


def render_bar_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = BAR_COLORS[: len(df)]
    bars = ax.bar(df[category_col], df[value_col], color=colors)
    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.set_ylim(0, max(df[value_col].max() + 1, 1))
    plt.xticks(rotation=25, ha="right")

    for bar, value in zip(bars, df[value_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


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

st.success(
    f"მონაცემები ავტომატურად ჩაიტვირთა: {len(model.candidates)} კანდიდატი, {len(model.electors)} ამომრჩეველი."
)

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
        candidate_absence_pct = st.slider("კანდიდატის გაცდენის შანსი (%)", min_value=0, max_value=100, value=2)
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
col1, col2, col3 = st.columns(3)
col1.metric("საბოლოო შედეგი", single_result["winner"])
col2.metric("დასწრება", single_result["attendance_count"])
col3.metric("გაცდენა", single_result["absence_count"])

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
left, right = st.columns([1, 1])
with left:
    st.dataframe(first_vote_df, use_container_width=True, hide_index=True)
with right:
    render_bar_chart(first_vote_df, "კანდიდატი", "ხმები", "პირველი ტურის ხმები")

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
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(runoff_df, use_container_width=True, hide_index=True)
        with right:
            render_bar_chart(runoff_df, "კანდიდატი", "ხმები", "მეორე ტურის ხმები")

st.subheader("საბოლოო გამარჯვების ალბათობები")
probabilities_df = monte_carlo_result["probabilities_df"]
metrics_cols = st.columns(len(probabilities_df))
for col, (_, row) in zip(metrics_cols, probabilities_df.iterrows()):
    col.metric(row["შედეგი"], f"{row['მოგების ალბათობა (%)']:.2f}%")

st.dataframe(probabilities_df, use_container_width=True, hide_index=True)
render_bar_chart(probabilities_df, "შედეგი", "მოგების ალბათობა (%)", "საბოლოო გამარჯვების ალბათობები")

st.subheader("Monte Carlo დამატებითი მაჩვენებლები")
mc1, mc2 = st.columns(2)
mc1.metric("საშუალო დასწრება", monte_carlo_result["avg_attendance"])
mc2.metric("მეორე ტურის დანიშვნის სიხშირე", f"{monte_carlo_result['runoff_rate_pct']:.2f}%")

st.markdown("### პირველივე ტურში გამარჯვების ალბათობა")
first_round_win_df = monte_carlo_result["first_round_win_df"]
st.dataframe(first_round_win_df, use_container_width=True, hide_index=True)
render_bar_chart(first_round_win_df, "კანდიდატი", "პირველივე ტურში გამარჯვება (%)", "პირველივე ტურში გამარჯვების ალბათობა")

st.markdown("### პირველი ტურის საშუალო ხმები და მეორე ტურში გასვლის სიხშირე")
avg_votes_df = monte_carlo_result["average_votes_df"]
st.dataframe(avg_votes_df, use_container_width=True, hide_index=True)
render_bar_chart(avg_votes_df, "კანდიდატი", "პირველი ტურის საშუალო ხმები", "პირველი ტურის საშუალო ხმები")

with st.expander("პირველი ტურის სრული ცხრილი"):
    st.dataframe(first_round["details_df"], use_container_width=True, hide_index=True)

if single_result["runoff_required"] and single_result["runoff"] is not None:
    with st.expander("მეორე ტურის სრული ცხრილი"):
        st.dataframe(single_result["runoff"]["details_df"], use_container_width=True, hide_index=True)

with st.expander("პრეფერენციების მატრიცა"):
    st.dataframe(build_preference_matrix(model), use_container_width=True, hide_index=True)
