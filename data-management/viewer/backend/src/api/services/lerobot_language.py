"""
Plan LeRobot language annotation rows onto an edited episode's output frames.

LeRobot 0.6 datasets keep language annotations in two per-frame list columns. ``language_persistent``
repeats an episode's rows on every frame; each row stays active from its ``timestamp`` until a later
row of the same style takes over. ``language_events`` holds rows that fire on the frame they sit on.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from .episode_edits import PlannedFrame, output_indices

LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"
LANGUAGE_COLUMNS = (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS)

_TIMESTAMP_TOLERANCE_S = 1e-4
"""Matches LeRobot's frame-timestamp tolerance, so float32 rounding can't push a row onto the next frame."""

LanguageRow = dict[str, Any]


def _sort_key(row: LanguageRow) -> tuple[float, str, str]:
    return float(row["timestamp"]), row.get("style") or "", row.get("role") or ""


def plan_persistent_rows(
    rows: list[LanguageRow],
    source_timestamps: list[float],
    plan: list[PlannedFrame],
    output_timestamps: list[float],
) -> list[LanguageRow]:
    """
    Move persistent rows from the source timeline onto the output frames.

    A row starts on the first source frame at or after its timestamp, or on the next kept frame when
    that frame was removed; a row with no later kept frame is dropped. When removals collapse rows of
    one style, role and camera onto the same output frame, only the rows from the latest source time
    stay, since those are the ones active at that frame.
    """
    positions = output_indices(plan)
    kept = sorted(positions)
    placed: dict[tuple[Any, ...], tuple[float, list[LanguageRow]]] = {}
    for row in rows:
        source_time = float(row["timestamp"])
        frame = bisect_left(source_timestamps, source_time - _TIMESTAMP_TOLERANCE_S)
        index = bisect_left(kept, frame)
        if index == len(kept):
            continue
        key = (row.get("style"), row.get("role"), row.get("camera"), positions[kept[index]])
        current = placed.get(key)
        if current is None or source_time > current[0]:
            placed[key] = (source_time, [row])
        elif source_time == current[0]:
            current[1].append(row)
    planned = [{**row, "timestamp": output_timestamps[key[-1]]} for key, (_, group) in placed.items() for row in group]
    return sorted(planned, key=_sort_key)


def plan_events(events_by_frame: list[list[LanguageRow] | None], plan: list[PlannedFrame]) -> list[list[LanguageRow]]:
    """Keep each kept frame's events and give inserted frames none, so no event fires twice."""
    return [(events_by_frame[frame.source] or []) if frame.following is None else [] for frame in plan]
