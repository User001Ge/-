from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional

import pandas as pd

from data_loader import ElectionModel

SECOND_ROUND_LABEL = "მეორე ტური"


@dataclass(frozen=True)
class SimulationParameters:
    selected_candidates: List[str]
    volatility_level: int = 3
    elector_absence_probability: float = 0.05
    candidate_absence_probability: float = 0.02



def _leftmost_max_choice(scores: Dict[str, int], ordered_candidates: List[str]) -> str:
    best_name = ordered_candidates[0]
    best_score = scores[best_name]
    for candidate in ordered_candidates[1:]:
        score = scores[candidate]
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name



def run_single_simulation(
    model: ElectionModel,
    params: SimulationParameters,
    rng: Optional[Random] = None,
) -> Dict[str, object]:
    model.validate_three_candidates(params.selected_candidates)
    rng = rng or Random()

    absent_candidates = {
        candidate
        for candidate in params.selected_candidates
        if rng.random() < params.candidate_absence_probability
    }

    vote_totals = {candidate: 0 for candidate in params.selected_candidates}
    row_details = []
    attendance_count = 0

    for elector in model.electors:
        is_absent = rng.random() < params.elector_absence_probability
        if is_absent or elector in absent_candidates:
            row_details.append(
                {
                    "მღვდელმთავარი": elector,
                    "დასწრება": "არა",
                    "არჩევანი": "—",
                    **{candidate: None for candidate in params.selected_candidates},
                }
            )
            continue

        attendance_count += 1
        base_scores = {
            candidate: model.preferences[elector][candidate]
            for candidate in params.selected_candidates
        }
        noisy_scores = {
            candidate: base_scores[candidate] + rng.randint(-params.volatility_level, params.volatility_level)
            for candidate in params.selected_candidates
        }
        chosen_candidate = _leftmost_max_choice(noisy_scores, params.selected_candidates)
        vote_totals[chosen_candidate] += 1

        row_details.append(
            {
                "მღვდელმთავარი": elector,
                "დასწრება": "კი",
                "არჩევანი": chosen_candidate,
                **{candidate: noisy_scores[candidate] for candidate in params.selected_candidates},
            }
        )

    max_votes = max(vote_totals.values())
    winners = [candidate for candidate, votes in vote_totals.items() if votes == max_votes]
    winner = SECOND_ROUND_LABEL if len(winners) > 1 else winners[0]

    return {
        "winner": winner,
        "vote_totals": vote_totals,
        "attendance_count": attendance_count,
        "absence_count": len(model.electors) - attendance_count,
        "absent_candidates": sorted(absent_candidates),
        "details_df": pd.DataFrame(row_details),
    }



def run_monte_carlo(
    model: ElectionModel,
    params: SimulationParameters,
    iterations: int,
    rng_seed: Optional[int] = None,
) -> Dict[str, object]:
    if iterations < 1:
        raise ValueError("იტერაციების რაოდენობა მინიმუმ 1 უნდა იყოს.")

    rng = Random(rng_seed)
    win_counts = {candidate: 0 for candidate in params.selected_candidates}
    win_counts[SECOND_ROUND_LABEL] = 0
    cumulative_votes = {candidate: 0 for candidate in params.selected_candidates}
    total_attendance = 0

    for _ in range(iterations):
        result = run_single_simulation(model, params, rng)
        win_counts[result["winner"]] += 1
        total_attendance += int(result["attendance_count"])
        for candidate, votes in result["vote_totals"].items():
            cumulative_votes[candidate] += int(votes)

    probability_rows = []
    for name, count in win_counts.items():
        probability_rows.append(
            {
                "შედეგი": name,
                "გამარჯვებების რაოდენობა": count,
                "მოგების ალბათობა (%)": round((count / iterations) * 100, 2),
            }
        )

    average_vote_rows = []
    for candidate, total_votes in cumulative_votes.items():
        average_vote_rows.append(
            {
                "კანდიდატი": candidate,
                "საშუალო ხმები": round(total_votes / iterations, 2),
            }
        )

    return {
        "probabilities_df": pd.DataFrame(probability_rows),
        "average_votes_df": pd.DataFrame(average_vote_rows),
        "avg_attendance": round(total_attendance / iterations, 2),
    }
