"""Tests for exporting LeRobot v3.0 episodes as a derived LeRobot v3.0 dataset."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.api.models.datasources import FrameInsertion
from src.api.services.episode_edits import EpisodeEditOperations, SubtaskSegment, TrajectoryAdjustment
from src.api.services.image_transform import CropRegion, ImageTransform, ResizeDimensions
from src.api.services.lerobot_exporter import ADJUSTED_STATE, ADJUSTED_STATE_MASK, PROVENANCE_FILE, LeRobotExporter

FPS = 10
WIDTH, HEIGHT = 32, 24
CAMERA = "observation.images.front"
LENGTHS = (12, 8)
STATS_KEYS = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}


def _gray(episode: int, frame: int) -> int:
    return 40 + 15 * frame if episode == 0 else 200 - 15 * frame


def _state(episode: int, frame: int) -> list[float]:
    return [float(episode), float(frame), float(episode * 100 + frame)]


def _write_source(root: Path, vector: pa.DataType | None = None) -> Path:
    """Write a two-episode v3.0 dataset whose frames and rows encode their episode and frame."""
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    video = root / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=FPS, options={"g": "2", "crf": "18"})
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for episode, length in enumerate(LENGTHS):
            for frame in range(length):
                image = np.full((HEIGHT, WIDTH, 3), _gray(episode, frame), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    rows: dict[str, list[Any]] = {name: [] for name in ("state", "phase", "flag", "timestamp", "frame", "episode")}
    episodes = []
    offset = 0
    for episode, length in enumerate(LENGTHS):
        for frame in range(length):
            rows["state"].append(_state(episode, frame))
            rows["phase"].append(frame // 4)
            rows["flag"].append(frame % 2 == 0)
            rows["timestamp"].append(frame / FPS)
            rows["frame"].append(frame)
            rows["episode"].append(episode)
        episodes.append(
            {
                "episode_index": episode,
                "tasks": ["pick the part"],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
                f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": 0,
                f"videos/{CAMERA}/from_timestamp": offset / FPS,
                f"videos/{CAMERA}/to_timestamp": (offset + length) / FPS,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "stats/observation.state/count": [length],
            }
        )
        offset += length
    vector = vector or pa.list_(pa.float32(), 3)
    total = len(rows["frame"])
    data = pa.table(
        {
            "observation.state": pa.array(rows["state"], type=vector),
            "action": pa.array([[2 * value for value in state] for state in rows["state"]], type=vector),
            "observation.phase": pa.array(rows["phase"], type=pa.int64()),
            "observation.flag": pa.array(rows["flag"], type=pa.bool_()),
            "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
            "frame_index": pa.array(rows["frame"], type=pa.int64()),
            "episode_index": pa.array(rows["episode"], type=pa.int64()),
            "index": pa.array(range(total), type=pa.int64()),
            "task_index": pa.array([0] * total, type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["pick the part"]}), root / "meta/tasks.parquet")
    scalar = {"shape": [1], "names": None}
    names = ["x", "y", "z"]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "fixture",
        "total_episodes": len(LENGTHS),
        "total_frames": total,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(LENGTHS)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAMERA: {
                "dtype": "video",
                "shape": [HEIGHT, WIDTH, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": HEIGHT,
                    "video.width": WIDTH,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": FPS,
                    "video.channels": 3,
                    "video.g": 2,
                    "video.crf": 18,
                    "has_audio": False,
                },
            },
            "observation.state": {"dtype": "float32", "shape": [3], "names": names},
            "action": {"dtype": "float32", "shape": [3], "names": names},
            "observation.phase": {"dtype": "int64", **scalar},
            "observation.flag": {"dtype": "bool", **scalar},
            "timestamp": {"dtype": "float32", **scalar},
            "frame_index": {"dtype": "int64", **scalar},
            "episode_index": {"dtype": "int64", **scalar},
            "index": {"dtype": "int64", **scalar},
            "task_index": {"dtype": "int64", **scalar},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _write_source(tmp_path / "datasets/capture/lerobot")


def _export(source: Path, episodes: list[int], edits: dict[int, EpisodeEditOperations] | None = None) -> Path:
    output = source.parents[1] / "edited"
    result = LeRobotExporter(source, output, dataset_id="capture--lerobot").export_episodes(episodes, edits)
    assert result.success, result.error
    assert result.output_files == [str(output)]
    return output


def _edits(episode: int, **operations: Any) -> dict[int, EpisodeEditOperations]:
    return {episode: EpisodeEditOperations(dataset_id="capture--lerobot", episode_index=episode, **operations)}


def _decode(path: Path) -> list[tuple[float, np.ndarray]]:
    with av.open(str(path)) as container:
        return [(frame.time, frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]


def _video(output: Path) -> Path:
    return output / f"videos/{CAMERA}/chunk-000/file-000.mp4"


def _info(output: Path) -> dict[str, Any]:
    return json.loads((output / "meta/info.json").read_text())


def _episodes(output: Path) -> list[dict[str, Any]]:
    return pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()


def _data(output: Path) -> pa.Table:
    return pq.read_table(output / "data/chunk-000/file-000.parquet")


def _digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_removed_frames_renumber_the_timeline_and_leave_the_source_unchanged(source: Path) -> None:
    before = _digests(source)

    output = _export(source, [0], _edits(0, removed_frames={2, 3, 50}))

    kept = [0, 1, *range(4, 12)]
    data = _data(output)
    assert data.column("frame_index").to_pylist() == list(range(10))
    assert data.column("index").to_pylist() == list(range(10))
    assert data.column("episode_index").to_pylist() == [0] * 10
    np.testing.assert_allclose(data.column("timestamp").to_numpy(), np.arange(10) / FPS, atol=1e-6)
    assert data.column("observation.state").to_pylist() == [_state(0, frame) for frame in kept]
    frames = _decode(_video(output))
    assert len(frames) == 10
    for (time, image), (position, frame) in zip(frames, enumerate(kept), strict=True):
        assert abs(time - position / FPS) < 1e-4
        assert abs(float(image.mean()) - _gray(0, frame)) < 4
    info = _info(output)
    assert (info["total_episodes"], info["total_frames"], info["splits"]) == (1, 10, {"train": "0:1"})
    provenance = json.loads((output / PROVENANCE_FILE).read_text())
    assert provenance["source"] == {"dataset_id": "capture--lerobot", "codebase_version": "v3.0", "fps": FPS}
    assert provenance["episodes"][0]["frame_sources"] == kept
    assert provenance["episodes"][0]["edits"]["removed_frames"] == [2, 3]
    assert _digests(source) == before


def test_two_episode_exports_carry_offsets_ranges_and_videos(source: Path) -> None:
    output = _export(source, [1, 0], _edits(1, removed_frames={0}))

    episodes = _episodes(output)
    assert [episode["episode_index"] for episode in episodes] == [0, 1]
    assert [(episode["dataset_from_index"], episode["dataset_to_index"]) for episode in episodes] == [(0, 7), (7, 19)]
    windows = [
        (episode[f"videos/{CAMERA}/from_timestamp"], episode[f"videos/{CAMERA}/to_timestamp"]) for episode in episodes
    ]
    assert windows == [(0.0, 0.7), (0.7, 1.9)]
    assert all(episode["tasks"] == ["pick the part"] for episode in episodes)
    data = _data(output)
    assert data.column("index").to_pylist() == list(range(19))
    assert data.column("episode_index").to_pylist() == [0] * 7 + [1] * 12
    assert data.column("frame_index").to_pylist() == list(range(7)) + list(range(12))
    expected = [(1, frame) for frame in range(1, 8)] + [(0, frame) for frame in range(12)]
    assert data.column("observation.state").to_pylist() == [_state(*pair) for pair in expected]
    frames = _decode(_video(output))
    assert len(frames) == 19
    for (episode, (start, _end)), first in zip(enumerate(windows), (0, 7), strict=True):
        time, image = frames[first]
        assert abs(time - start) < 1e-4
        assert abs(float(image.mean()) - _gray(*expected[first])) < 4, episode
    for (_time, image), pair in zip(frames, expected, strict=True):
        assert abs(float(image.mean()) - _gray(*pair)) < 4


def test_inserted_frames_interpolate_floats_and_hold_other_features(source: Path) -> None:
    insertion = [FrameInsertion(after_frame_index=4, interpolation_factor=0.25)]

    output = _export(source, [0], _edits(0, inserted_frames=insertion, removed_frames={5}))

    data = _data(output)
    assert data.num_rows == 12
    inserted = data.slice(5, 1).to_pylist()[0]
    np.testing.assert_allclose(
        inserted["observation.state"], 0.75 * np.array(_state(0, 4)) + 0.25 * np.array(_state(0, 6))
    )
    assert inserted["observation.phase"] == 1
    assert inserted["observation.flag"] is True
    _time, image = _decode(_video(output))[5]
    assert abs(float(image.mean()) - (0.75 * _gray(0, 4) + 0.25 * _gray(0, 6))) < 4
    provenance = json.loads((output / PROVENANCE_FILE).read_text())["episodes"][0]
    assert provenance["frame_sources"][4:7] == [4, None, 6]
    assert provenance["edits"]["inserted_frames"] == [{"after_frame_index": 4, "interpolation_factor": 0.25}]


def test_crop_and_resize_change_the_camera_video_and_feature_shape(source: Path) -> None:
    transform = ImageTransform(
        crop=CropRegion(x=4, y=2, width=16, height=12), resize=ResizeDimensions(width=8, height=6)
    )

    output = _export(source, [0], _edits(0, camera_transforms={CAMERA: transform}))

    feature = _info(output)["features"][CAMERA]
    assert feature["shape"] == [6, 8, 3]
    assert (feature["info"]["video.height"], feature["info"]["video.width"]) == (6, 8)
    frames = _decode(_video(output))
    assert len(frames) == 12
    assert frames[0][1].shape == (6, 8, 3)


def test_trajectory_adjustments_add_derived_state_beside_the_recorded_state(source: Path) -> None:
    adjustments = [
        TrajectoryAdjustment(frame_index=2, channel_deltas={0: 0.5}),
        TrajectoryAdjustment(frame_index=7, channel_values={2: -1.0}),
    ]

    output = _export(source, [0], _edits(0, trajectory_adjustments=adjustments, removed_frames={1}))

    data = _data(output)
    recorded = np.array(data.column("observation.state").to_pylist())
    adjusted = np.array(data.column(ADJUSTED_STATE).to_pylist())
    mask = np.array(data.column(ADJUSTED_STATE_MASK).to_pylist())
    kept = [0, *range(2, 12)]
    np.testing.assert_allclose(recorded, [_state(0, frame) for frame in kept])
    assert mask.tolist() == [frame in {2, 7} for frame in kept]
    np.testing.assert_allclose(adjusted[~mask], recorded[~mask])
    np.testing.assert_allclose(adjusted[1], [0.5, 2.0, 2.0])
    np.testing.assert_allclose(adjusted[6], [0.0, 7.0, -1.0])
    features = _info(output)["features"]
    assert features[ADJUSTED_STATE] == {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]}
    assert features[ADJUSTED_STATE_MASK] == {"dtype": "bool", "shape": [1], "names": None}
    edits = json.loads((output / PROVENANCE_FILE).read_text())["episodes"][0]["edits"]
    assert [adjustment["frame_index"] for adjustment in edits["trajectory_adjustments"]] == [2, 7]


def test_episode_and_dataset_stats_match_the_exported_data(source: Path) -> None:
    output = _export(source, [0, 1], _edits(0, removed_frames={3}))

    data = _data(output)
    episodes = _episodes(output)
    stats = json.loads((output / "meta/stats.json").read_text())
    numeric = [name for name, feature in _info(output)["features"].items() if feature["dtype"] != "video"]
    assert set(stats) == {*numeric, CAMERA}
    for name in numeric:
        column = data.column(name).to_pylist()
        values = np.array([value if isinstance(value, list) else [value] for value in column], dtype=np.float64)
        assert set(stats[name]) == STATS_KEYS
        np.testing.assert_allclose(stats[name]["mean"], values.mean(axis=0))
        np.testing.assert_allclose(stats[name]["q90"], np.quantile(values, 0.9, axis=0))
        assert stats[name]["count"] == [len(values)]
        for episode in episodes:
            rows = values[episode["dataset_from_index"] : episode["dataset_to_index"]]
            np.testing.assert_allclose(episode[f"stats/{name}/min"], rows.min(axis=0))
            np.testing.assert_allclose(episode[f"stats/{name}/std"], rows.std(axis=0))
            assert episode[f"stats/{name}/count"] == [len(rows)]
    frames = np.stack([image for _time, image in _decode(_video(output))]).astype(np.float64) / 255
    assert set(stats[CAMERA]) == STATS_KEYS
    assert np.array(stats[CAMERA]["mean"]).shape == (3, 1, 1)
    np.testing.assert_allclose(np.array(stats[CAMERA]["mean"]).ravel(), frames.mean(axis=(0, 1, 2)), atol=0.02)
    assert stats[CAMERA]["count"] == [19]
    for episode in episodes:
        assert np.array(episode[f"stats/{CAMERA}/max"]).shape == (3, 1, 1)
        assert episode[f"stats/{CAMERA}/count"] == [episode["length"]]


def test_subtasks_are_remapped_to_output_frames_in_provenance(source: Path) -> None:
    subtasks = [
        SubtaskSegment(id="reach", label="Reach", frame_range=(0, 4), color="#ff0000", source="manual"),
        SubtaskSegment(id="grasp", label="Grasp", frame_range=(5, 11), color="#00ff00", source="manual"),
        SubtaskSegment(id="gone", label="Gone", frame_range=(2, 3), color="#0000ff", source="auto"),
    ]
    insertion = [FrameInsertion(after_frame_index=1, interpolation_factor=0.5)]

    output = _export(source, [0], _edits(0, subtasks=subtasks, inserted_frames=insertion, removed_frames={3}))

    edits = json.loads((output / PROVENANCE_FILE).read_text())["episodes"][0]["edits"]
    assert [(subtask["id"], subtask["frame_range"]) for subtask in edits["subtasks"]] == [
        ("reach", [0, 4]),
        ("grasp", [5, 11]),
    ]


def test_writes_use_standard_paths_even_when_source_templates_point_elsewhere(source: Path) -> None:
    info = json.loads((source / "meta/info.json").read_text())
    info["data_path"] = "../lerobot/data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    (source / "meta/info.json").write_text(json.dumps(info))

    output = _export(source, [0])

    assert (output / "data/chunk-000/file-000.parquet").is_file()
    assert not (source.parents[1] / "lerobot").exists()
    assert _info(output)["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def test_variable_length_vectors_interpolate_and_keep_their_list_type(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "datasets/capture/lerobot", vector=pa.list_(pa.float32()))
    insertion = [FrameInsertion(after_frame_index=4, interpolation_factor=0.5)]

    output = _export(source, [0], _edits(0, inserted_frames=insertion))

    data = _data(output)
    assert data.schema.field("observation.state").type == pa.list_(pa.float32())
    expected = 0.5 * np.array(_state(0, 4)) + 0.5 * np.array(_state(0, 5))
    np.testing.assert_allclose(data.column("observation.state")[5].as_py(), expected)
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["observation.state"]["count"] == [13]
    assert len(stats["observation.state"]["mean"]) == 3


def test_unrecorded_encoder_settings_fall_back_to_lerobot_defaults(source: Path) -> None:
    info = json.loads((source / "meta/info.json").read_text())
    for key in ("video.g", "video.crf"):
        del info["features"][CAMERA]["info"][key]
    (source / "meta/info.json").write_text(json.dumps(info))

    output = _export(source, [0])

    video_info = _info(output)["features"][CAMERA]["info"]
    assert (video_info["video.g"], video_info["video.crf"]) == (2, 30)
    with av.open(str(_video(output))) as container:
        keyframes = sum(frame.key_frame for frame in container.decode(video=0))
    assert keyframes >= 6


def test_export_root_mode_follows_the_umask_or_the_existing_directory(source: Path, tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe.mkdir()

    output = _export(source, [0])

    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(probe.stat().st_mode)
    existing = source.parents[1] / "existing"
    existing.mkdir()
    existing.chmod(0o750)
    assert LeRobotExporter(source, existing).export_episodes([0]).success
    assert stat.S_IMODE(existing.stat().st_mode) == 0o750


def _break_version(source: Path) -> None:
    info = json.loads((source / "meta/info.json").read_text())
    (source / "meta/info.json").write_text(json.dumps({**info, "codebase_version": "v2.1"}))


def _escape_camera_key(source: Path) -> None:
    info = json.loads((source / "meta/info.json").read_text())
    info["features"]["../escape"] = info["features"].pop(CAMERA)
    (source / "meta/info.json").write_text(json.dumps(info))


def _shorten_video_window(source: Path) -> None:
    path = source / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][f"videos/{CAMERA}/to_timestamp"] = 1.0
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.mark.parametrize(
    ("prepare", "episodes", "edits", "message"),
    [
        (_break_version, [0], None, "supports v3.0 datasets, not v2.1"),
        (
            None,
            [0, 1],
            _edits(0, global_transform=ImageTransform(resize=ResizeDimensions(width=16, height=12))),
            "different sizes",
        ),
        (
            None,
            [0],
            _edits(0, global_transform=ImageTransform(resize=ResizeDimensions(width=15, height=12))),
            "even width",
        ),
        (_shorten_video_window, [0], None, "video ends after 10 frames, before frame 10"),
        (
            _shorten_video_window,
            [0],
            _edits(0, removed_frames={10, 11}),
            "video window holds 10 frames for 12 data rows",
        ),
        (None, [0, 0], None, "only once"),
        (_escape_camera_key, [0], None, "is not a plain name"),
    ],
)
def test_failed_exports_leave_nothing_behind(
    source: Path, prepare: Any, episodes: list[int], edits: dict[int, EpisodeEditOperations] | None, message: str
) -> None:
    if prepare is not None:
        prepare(source)
    parent = source.parents[1]
    before = sorted(path.name for path in parent.iterdir())

    result = LeRobotExporter(source, parent / "edited").export_episodes(episodes, edits)

    assert result.success is False
    assert message in result.error
    assert sorted(path.name for path in parent.iterdir()) == before


def test_a_non_empty_output_directory_is_refused(source: Path) -> None:
    output = source.parents[1] / "edited"
    output.mkdir()
    (output / "keep.txt").write_text("existing")

    result = LeRobotExporter(source, output).export_episodes([0])

    assert result.success is False
    assert "new or empty" in result.error
    assert [path.name for path in output.iterdir()] == ["keep.txt"]
