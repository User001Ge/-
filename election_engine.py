from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional

import pandas as pd

from data_loader import ElectionModel

RUNOFF_UNCLEAR_LABEL = "მეორე ტურის შემადგენლობა გაურკვეველია"


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


def _majority_threshold(valid_votes: int) -> int:
    return (valid_votes // 2) + 1


def _simulate_round(
    model: ElectionModel,
    candidates: List[str],
    participating_electors: List[str],
    params: SimulationParameters,
    rng: Random,
) -> Dict[str, object]:
    vote_totals = {candidate: 0 for candidate in candidates}
    row_details = []

    for elector in model.electors:
        if elector not in participating_electors:
            row_details.append(
                {
                    "მღვდელმთავარი": elector,
                    "დასწრება": "არა",
                    "არჩევანი": "—",
                    **{candidate: None for candidate in candidates},
                }
            )
            continue

        base_scores = {candidate: model.preferences[elector][candidate] for candidate in candidates}
        noisy_scores = {
            candidate: base_scores[candidate] + rng.randint(-params.volatility_level, params.volatility_level)
            for candidate in candidates
        }
        chosen_candidate = _leftmost_max_choice(noisy_scores, candidates)
        vote_totals[chosen_candidate] += 1

        row_details.append(
            {
                "მღვდელმთავარი": elector,
                "დასწრება": "კი",
                "არჩევანი": chosen_candidate,
                **{candidate: noisy_scores[candidate] for candidate in candidates},
            }
        )

    valid_votes = len(participating_electors)
    threshold = _majority_threshold(valid_votes) if valid_votes > 0 else 1

    return {
        "vote_totals": vote_totals,
        "valid_votes": valid_votes,
        "majority_threshold": threshold,
        "details_df": pd.DataFrame(row_details),
    }


def _resolve_runoff_candidates(first_round_totals: Dict[str, int], ordered_candidates: List[str]) -> Optional[List[str]]:
    groups: Dict[int, List[str]] = {}
    for candidate in ordered_candidates:
        groups.setdefault(first_round_totals[candidate], []).append(candidate)

    sorted_groups = sorted(groups.items(), key=lambda item: item[0], reverse=True)
    top_group = sorted_groups[0][1]

    if len(top_group) >= 3:
        return None

    if len(top_group) == 2:
        return top_group

    if len(sorted_groups) < 2:
        return None

    second_group = sorted_groups[1][1]
    if len(second_group) != 1:
        return None

    return [top_group[0], second_group[0]]


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

    participating_electors: List[str] = []
    absent_electors: List[str] = []

    for elector in model.electors:
        is_absent = rng.random() < params.elector_absence_probability
        if is_absent or elector in absent_candidates:
            absent_electors.append(elector)
        else:
            participating_electors.append(elector)

    first_round = _simulate_round(model, params.selected_candidates, participating_electors, params, rng)
    first_round_totals = first_round["vote_totals"]
    valid_votes = int(first_round["valid_votes"])
    threshold = int(first_round["majority_threshold"])

    first_round_winner = None
    for candidate in params.selected_candidates:
        if first_round_totals[candidate] >= threshold:
            first_round_winner = candidate
            break

    runoff_required = first_round_winner is None
    runoff_candidates: Optional[List[str]] = None
    runoff_result: Optional[Dict[str, object]] = None
    final_winner: str

    if not runoff_required:
        final_winner = first_round_winner
    else:
        runoff_candidates = _resolve_runoff_candidates(first_round_totals, params.selected_candidates)
        if runoff_candidates is None:
            final_winner = RUNOFF_UNCLEAR_LABEL
        else:
            runoff_result = _simulate_round(model, runoff_candidates, participating_electors, params, rng)
            runoff_totals = runoff_result["vote_totals"]
            max_votes = max(runoff_totals.values())
            winners = [c for c in runoff_candidates if runoff_totals[c] == max_votes]
            final_winner = winners[0] if len(winners) == 1 else RUNOFF_UNCLEAR_LABEL

    return {
        "winner": final_winner,
        "first_round": {
            "vote_totals": first_round_totals,
            "valid_votes": valid_votes,
            "majority_threshold": threshold,
            "details_df": first_round["details_df"],
        },
        "runoff_required": runoff_required,
        "runoff_candidates": runoff_candidates,
        "runoff": runoff_result,
        "attendance_count": len(participating_electors),
        "absence_count": len(absent_electors),
        "absent_candidates": sorted(absent_candidates),
        "participating_electors": participating_electors,
        "absent_electors": absent_electors,
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
    final_result_counts = {candidate: 0 for candidate in params.selected_candidates}
    final_result_counts[RUNOFF_UNCLEAR_LABEL] = 0

    first_round_win_counts = {candidate: 0 for candidate in params.selected_candidates}
    runoff_count = 0
    cumulative_first_round_votes = {candidate: 0 for candidate in params.selected_candidates}
    cumulative_runoff_appearances = {candidate: 0 for candidate in params.selected_candidates}
    total_attendance = 0

    for _ in range(iterations):
        result = run_single_simulation(model, params, rng)
        final_result_counts[result["winner"]] += 1
        total_attendance += int(result["attendance_count"])

        first_round = result["first_round"]
        for candidate, votes in first_round["vote_totals"].items():
            cumulative_first_round_votes[candidate] += int(votes)
            if int(votes) >= int(first_round["majority_threshold"]):
                first_round_win_counts[candidate] += 1

        if result["runoff_required"]:
            runoff_count += 1
            if result["runoff_candidates"]:
                for candidate in result["runoff_candidates"]:
                    cumulative_runoff_appearances[candidate] += 1

    final_probability_rows = [
        {
            "შედეგი": name,
            "გამარჯვებების რაოდენობა": count,
            "მოგების ალბათობა (%)": round((count / iterations) * 100, 2),
        }
        for name, count in final_result_counts.items()
    ]

    first_round_win_rows = [
        {
            "კანდიდატი": candidate,
            "პირველივე ტურში გამარჯვება (%)": round((count / iterations) * 100, 2),
        }
        for candidate, count in first_round_win_counts.items()
    ]

    average_vote_rows = [
        {
            "კანდიდატი": candidate,
            "პირველი ტურის საშუალო ხმები": round(total_votes / iterations, 2),
            "მეორე ტურში გასვლის სიხშირე (%)": round((cumulative_runoff_appearances[candidate] / iterations) * 100, 2),
        }
        for candidate, total_votes in cumulative_first_round_votes.items()
    ]

    return {
        "probabilities_df": pd.DataFrame(final_probability_rows),
        "first_round_win_df": pd.DataFrame(first_round_win_rows),
        "average_votes_df": pd.DataFrame(average_vote_rows),
        "avg_attendance": round(total_attendance / iterations, 2),
        "runoff_rate_pct": round((runoff_count / iterations) * 100, 2),
    }
