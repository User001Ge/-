from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


@dataclass(frozen=True)
class ElectionModel:
    candidates: List[str]
    electors: List[str]
    preferences: Dict[str, Dict[str, int]]

    def validate_three_candidates(self, selected_candidates: List[str]) -> None:
        if len(selected_candidates) != 3:
            raise ValueError("ზუსტად 3 კანდიდატი უნდა იყოს არჩეული.")
        if len(set(selected_candidates)) != 3:
            raise ValueError("სამივე კანდიდატი განსხვავებული უნდა იყოს.")
        missing = [name for name in selected_candidates if name not in self.candidates]
        if missing:
            raise ValueError(f"მონაცემებში ეს კანდიდატები ვერ მოიძებნა: {', '.join(missing)}")
        for elector in self.electors:
            missing_scores = [c for c in selected_candidates if c not in self.preferences[elector]]
            if missing_scores:
                raise ValueError(
                    f"ამომრჩეველს '{elector}' არ აქვს ქულა ამ კანდიდატებზე: {', '.join(missing_scores)}"
                )


class PreferenceFileError(ValueError):
    """Raised when the uploaded preference file has an invalid structure."""



def load_model_from_excel(path: str | Path) -> ElectionModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ფაილი ვერ მოიძებნა: {path}")

    df = pd.read_excel(path, sheet_name=0)
    return load_model_from_dataframe(df)



def load_model_from_dataframe(df: pd.DataFrame) -> ElectionModel:
    if df.empty:
        raise PreferenceFileError("Excel ფაილი ცარიელია.")

    first_col = str(df.columns[0]).strip()
    if not first_col:
        raise PreferenceFileError("პირველ სვეტს უნდა ჰქონდეს ამომრჩეველთა სათაური.")

    candidate_names = [clean_name(c) for c in df.columns[1:]]
    if len(candidate_names) < 3:
        raise PreferenceFileError("ფაილში მინიმუმ 3 კანდიდატი უნდა იყოს.")
    if len(candidate_names) != len(set(candidate_names)):
        raise PreferenceFileError("კანდიდატების სახელები დუბლირებულია.")

    electors: List[str] = []
    preferences: Dict[str, Dict[str, int]] = {}

    for row_index, row in df.iterrows():
        elector = clean_name(row.iloc[0])
        if not elector:
            raise PreferenceFileError(f"{row_index + 2}-ე მწკრივში ამომრჩევლის სახელი ცარიელია.")
        if elector in preferences:
            raise PreferenceFileError(f"ამომრჩეველი დუბლირებულია: {elector}")

        scores: Dict[str, int] = {}
        for candidate, raw_value in zip(candidate_names, row.iloc[1:]):
            if pd.isna(raw_value):
                raise PreferenceFileError(
                    f"ამომრჩეველს '{elector}' არ აქვს შევსებული ქულა კანდიდატზე '{candidate}'."
                )
            try:
                score = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise PreferenceFileError(
                    f"არასწორი ქულა ამომრჩეველთან '{elector}', კანდიდატზე '{candidate}': {raw_value!r}"
                ) from exc
            scores[candidate] = score

        electors.append(elector)
        preferences[elector] = scores

    return ElectionModel(candidates=candidate_names, electors=electors, preferences=preferences)



def clean_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())
