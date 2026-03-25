from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import PreferenceFileError, load_model_from_dataframe, load_model_from_excel
from election_engine import SECOND_ROUND_LABEL, SimulationParameters, run_monte_carlo, run_single_simulation

st.set_page_config(page_title="არჩევნების სიმულატორი", page_icon="🗳️", layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_model(file_bytes: bytes, suffix: str) -> object:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        return load_model_from_excel(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)



def build_preference_matrix(model) -> pd.DataFrame:
    rows = []
    for elector in model.electors:
        row = {"მღვდელმთავარი": elector}
        row.update(model.preferences[elector])
        rows.append(row)
    return pd.DataFrame(rows)


st.title("საარჩევნო სიმულატორი")
st.caption("Excel-ფაილიდან კითხულობს კანდიდატებს, ამომრჩევლებს და პრეფერენციების ქულებს. თითო არჩევნებში ირჩევა ზუსტად 3 კანდიდატი.")

uploaded_file = st.file_uploader(
    "ატვირთე პრეფერენციების Excel ფაილი",
    type=["xlsx", "xlsm", "xltx", "xltm"],
)

if not uploaded_file:
    st.info("აპლიკაციის დასაწყებად ატვირთე Excel ფაილი პრეფერენციების მატრიცით.")
    st.stop()

try:
    model = load_uploaded_model(uploaded_file.getvalue(), Path(uploaded_file.name).suffix or ".xlsx")
except PreferenceFileError as exc:
    st.error(f"ფაილის სტრუქტურის შეცდომა: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"ფაილის წაკითხვა ვერ მოხერხდა: {exc}")
    st.stop()

st.success(
    f"ფაილი წარმატებით ჩაიტვირთა: {len(model.candidates)} კანდიდატი, {len(model.electors)} ამომრჩეველი."
)

with st.sidebar:
    st.header("პარამეტრები")

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
col1.metric("შედეგი", single_result["winner"])
col2.metric("დასწრება", single_result["attendance_count"])
col3.metric("გაცდენა", single_result["absence_count"])

if single_result["absent_candidates"]:
    st.write("**გაცდენილი კანდიდატები:**", ", ".join(single_result["absent_candidates"]))
else:
    st.write("**გაცდენილი კანდიდატები:** არავინ")

vote_df = pd.DataFrame(
    {
        "კანდიდატი": list(single_result["vote_totals"].keys()),
        "ხმები": list(single_result["vote_totals"].values()),
    }
)

left, right = st.columns([1, 1])
with left:
    st.dataframe(vote_df, use_container_width=True, hide_index=True)
with right:
    st.bar_chart(vote_df.set_index("კანდიდატი"))

st.subheader("მოგების ალბათობები")
probabilities_df = monte_carlo_result["probabilities_df"]
metrics_cols = st.columns(len(probabilities_df))
for col, (_, row) in zip(metrics_cols, probabilities_df.iterrows()):
    col.metric(row["შედეგი"], f"{row['მოგების ალბათობა (%)']:.2f}%")

st.dataframe(probabilities_df, use_container_width=True, hide_index=True)
st.bar_chart(probabilities_df.set_index("შედეგი")[["მოგების ალბათობა (%)"]])

st.subheader("საშუალო ხმები Monte Carlo-ს მიხედვით")
avg_votes_df = monte_carlo_result["average_votes_df"]
st.metric("საშუალო დასწრება", monte_carlo_result["avg_attendance"])
st.dataframe(avg_votes_df, use_container_width=True, hide_index=True)
st.bar_chart(avg_votes_df.set_index("კანდიდატი"))

with st.expander("ერთი გაშვების სრული ცხრილი"):
    st.dataframe(single_result["details_df"], use_container_width=True, hide_index=True)

with st.expander("პრეფერენციების მატრიცა"):
    st.dataframe(build_preference_matrix(model), use_container_width=True, hide_index=True)
