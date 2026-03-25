from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import PreferenceFileError, load_model_from_excel
from election_engine import (
    RUNOFF_UNCLEAR_LABEL,
    SimulationParameters,
    run_monte_carlo,
    run_single_simulation,
)

st.set_page_config(page_title="არჩევნების სიმულატორი", page_icon="🗳️", layout="wide")

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
    selected_candidates = st.multiselect(
        "აირჩიე 3 კანდიდატი",
        options=model.candidates,
        default=default_candidates,
        max_selections=3,
    )

    iterations = st.number_input("იტერაციების რაოდენობა", min_value=1, max_value=200_000, value=10_000, step=500)
    volatility_level = st.slider("რყევადობის დონე", min_value=0, max_value=10, value=3)
    elector_absence_pct = st.slider("ამომრჩევლის გაცდენის შანსი (%)", min_value=0, max_value=100, value=5)
    candidate_absence_pct = st.slider("კანდიდატის გაცდენის შანსი (%)", min_value=0, max_value=100, value=2)
    rng_seed_text = st.text_input("Seed (სურვილისამებრ, გამეორებადი შედეგებისთვის)", value="")

if len(selected_candidates) != 3:
    st.warning("სიმულაციის გასაშვებად ზუსტად 3 კანდიდატი უნდა აირჩიო.")
    st.stop()

try:
    rng_seed = int(rng_seed_text) if rng_seed_text.strip() else None
except ValueError:
    st.error("Seed უნდა იყოს მთელი რიცხვი.")
    st.stop()

params = SimulationParameters(
    selected_candidates=selected_candidates,
    volatility_level=volatility_level,
    elector_absence_probability=elector_absence_pct / 100,
    candidate_absence_probability=candidate_absence_pct / 100,
)

try:
    single_result = run_single_simulation(model, params)
    monte_carlo_result = run_monte_carlo(model, params, int(iterations), rng_seed=rng_seed)
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
    st.bar_chart(first_vote_df.set_index("კანდიდატი"))

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
            st.bar_chart(runoff_df.set_index("კანდიდატი"))

st.subheader("საბოლოო გამარჯვების ალბათობები")
probabilities_df = monte_carlo_result["probabilities_df"]
metrics_cols = st.columns(len(probabilities_df))
for col, (_, row) in zip(metrics_cols, probabilities_df.iterrows()):
    col.metric(row["შედეგი"], f"{row['მოგების ალბათობა (%)']:.2f}%")

st.dataframe(probabilities_df, use_container_width=True, hide_index=True)
st.bar_chart(probabilities_df.set_index("შედეგი")[["მოგების ალბათობა (%)"]])

st.subheader("Monte Carlo დამატებითი მაჩვენებლები")
mc1, mc2 = st.columns(2)
mc1.metric("საშუალო დასწრება", monte_carlo_result["avg_attendance"])
mc2.metric("მეორე ტურის დანიშვნის სიხშირე", f"{monte_carlo_result['runoff_rate_pct']:.2f}%")

st.markdown("### პირველივე ტურში გამარჯვების ალბათობა")
first_round_win_df = monte_carlo_result["first_round_win_df"]
st.dataframe(first_round_win_df, use_container_width=True, hide_index=True)

st.markdown("### პირველი ტურის საშუალო ხმები და მეორე ტურში გასვლის სიხშირე")
avg_votes_df = monte_carlo_result["average_votes_df"]
st.dataframe(avg_votes_df, use_container_width=True, hide_index=True)
st.bar_chart(avg_votes_df.set_index("კანდიდატი")[["პირველი ტურის საშუალო ხმები"]])

with st.expander("პირველი ტურის სრული ცხრილი"):
    st.dataframe(first_round["details_df"], use_container_width=True, hide_index=True)

if single_result["runoff_required"] and single_result["runoff"] is not None:
    with st.expander("მეორე ტურის სრული ცხრილი"):
        st.dataframe(single_result["runoff"]["details_df"], use_container_width=True, hide_index=True)

with st.expander("პრეფერენციების მატრიცა"):
    st.dataframe(build_preference_matrix(model), use_container_width=True, hide_index=True)
