"""
LeRobot exporter: writes edited episodes of a LeRobot v3.0 dataset as a new v3.0 dataset.

The source dataset is never modified. Recorded features are copied unchanged, and trajectory
adjustments become the derived ``adjusted.observation.state`` and ``adjusted.observation.state_mask``
features. Removing or inserting frames renumbers the timeline so every timestamp stays
``frame_index / fps``; ``dataviewer-export.json`` records which source frame each output frame
came from, the applied edits and the remapped subtasks. Videos are decoded and re-encoded with
the source's recorded encoder settings, and per-episode and dataset statistics are recomputed.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from .episode_edits import (
    EpisodeEditOperations,
    ExportError,
    ExportProgress,
    ExportResult,
    PlannedFrame,
    ProgressCallback,
    apply_trajectory_adjustments,
    output_indices,
    plan_frames,
    remap_subtasks,
)
from .frame_interpolation import interpolate_image
from .image_transform import ImageTransform, ImageTransformError, apply_transform, get_output_dimensions
from .lerobot_loader import LeRobotDatasetInfo, LeRobotLoader, LeRobotLoaderError

STATE_FEATURE = "observation.state"
ADJUSTED_STATE = "adjusted.observation.state"
ADJUSTED_STATE_MASK = "adjusted.observation.state_mask"
PROVENANCE_FILE = "dataviewer-export.json"

_SUPPORTED_VERSION = "v3.0"
_TIMELINE = ("timestamp", "frame_index", "episode_index", "index")
_STATS_DTYPES = {"float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "bool"}
_ENCODERS = {
    "av1": "libsvtav1",
    "libsvtav1": "libsvtav1",
    "h264": "libx264",
    "libx264": "libx264",
    "hevc": "libx265",
    "libx265": "libx265",
}
_QUANTILES = (1, 10, 50, 90, 99)
_IMAGE_SAMPLES = 100
_IMAGE_STATS_SIZE = 150


class LeRobotExportError(ExportError):
    """Exception raised for LeRobot export failures."""


@dataclass
class _Episode:
    """One requested episode: its source rows, metadata record, edits and output frame plan."""

    source_index: int
    output_index: int
    table: pa.Table
    record: dict[str, Any]
    edits: EpisodeEditOperations | None
    plan: list[PlannedFrame]
    offset: int = 0
    windows: dict[str, tuple[float, float]] = field(default_factory=dict)
    image_samples: dict[str, list[NDArray[np.uint8]]] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.plan)

    def transform(self, camera: str) -> ImageTransform | None:
        if self.edits is None:
            return None
        return (self.edits.camera_transforms or {}).get(camera, self.edits.global_transform)


class _EpisodeFrames:
    """Decode one episode's frames in order from its time window of a source video."""

    def __init__(self, path: Path, window: tuple[float, float], fps: float) -> None:
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._start, self._end = window
        self._tolerance = 0.5 / fps
        self._frames = self._decode()
        self._decoded = 0
        self._cache: dict[int, NDArray[np.uint8]] = {}

    def _decode(self) -> Iterator[NDArray[np.uint8]]:
        if self._start > 0 and self._stream.time_base is not None:
            self._container.seek(int(self._start / self._stream.time_base), stream=self._stream, backward=True)
        for frame in self._container.decode(self._stream):
            if frame.time is None or frame.time < self._start - self._tolerance:
                continue
            if frame.time >= self._end - self._tolerance:
                return
            yield frame.to_ndarray(format="rgb24")

    def get(self, index: int, keep_from: int) -> NDArray[np.uint8]:
        """Return source frame ``index``, keeping only frames at or after ``keep_from`` in memory."""
        self._cache = {key: value for key, value in self._cache.items() if key >= keep_from}
        while self._decoded <= index:
            frame = next(self._frames, None)
            if frame is None:
                raise LeRobotExportError(f"video ends after {self._decoded} frames, before frame {index}")
            if self._decoded == index:
                self._cache[index] = frame
            self._decoded += 1
        return self._cache[index]

    def finish(self, length: int) -> None:
        """Fail unless the window holds exactly ``length`` frames."""
        remaining = sum(1 for _ in self._frames)
        if self._decoded + remaining != length:
            raise LeRobotExportError(f"video window holds {self._decoded + remaining} frames for {length} data rows")

    def close(self) -> None:
        self._container.close()


class _VideoOutput:
    """Encode one camera's output video with the source feature's recorded encoder settings."""

    def __init__(self, path: Path, video_info: dict[str, Any], size: tuple[int, int], fps: float) -> None:
        codec = str(video_info.get("video.codec", ""))
        encoder = _ENCODERS.get(codec)
        if encoder is None:
            raise LeRobotExportError(f"video codec {codec!r} cannot be re-encoded")
        options = {
            key: str(video_info[f"video.{key}"]) for key in ("g", "crf") if video_info.get(f"video.{key}") is not None
        }
        preset = video_info.get("video.preset")
        if preset is not None and isinstance(preset, int) == (encoder == "libsvtav1"):
            options["preset"] = str(preset)
        rate = Fraction(fps).limit_denominator(1001)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), "w")
        self._stream = self._container.add_stream(encoder, rate=rate, options=options)
        self._stream.width, self._stream.height = size
        self._stream.pix_fmt = str(video_info.get("video.pix_fmt", "yuv420p"))
        self._time_base = 1 / rate
        self.frames = 0

    def write(self, image: NDArray[np.uint8]) -> None:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self.frames
        frame.time_base = self._time_base
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _numeric_stats(values: NDArray[np.float64]) -> dict[str, Any]:
    matrix = values.reshape(len(values), -1)
    stats: dict[str, Any] = {
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
        "count": [len(matrix)],
    }
    stats.update({f"q{q:02d}": np.quantile(matrix, q / 100, axis=0).tolist() for q in _QUANTILES})
    return stats


def _image_stats(samples: list[NDArray[np.uint8]]) -> dict[str, Any]:
    """Per-channel stats as ``(3, 1, 1)`` lists over ``[0, 1]`` pixels of the sampled frames."""
    pixels = np.concatenate([sample.reshape(-1, sample.shape[-1]) for sample in samples]).astype(np.float64) / 255

    def channels(vector: NDArray[np.float64]) -> list[list[list[float]]]:
        return [[[float(value)]] for value in vector]

    stats: dict[str, Any] = {
        "min": channels(pixels.min(axis=0)),
        "max": channels(pixels.max(axis=0)),
        "mean": channels(pixels.mean(axis=0)),
        "std": channels(pixels.std(axis=0)),
        "count": [len(samples)],
    }
    stats.update({f"q{q:02d}": channels(np.quantile(pixels, q / 100, axis=0)) for q in _QUANTILES})
    return stats


def _matrix(array: pa.Array) -> NDArray[np.float64]:
    """Return a numeric column as a ``(rows, width)`` float64 matrix."""
    if pa.types.is_fixed_size_list(array.type):
        values = array.flatten().to_numpy(zero_copy_only=False)
        return values.astype(np.float64).reshape(len(array), array.type.list_size)
    return array.to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(array), 1)


def _from_matrix(matrix: NDArray[np.float64], data_type: pa.DataType) -> pa.Array:
    """Rebuild a column of ``data_type`` from a ``(rows, width)`` matrix."""
    if pa.types.is_fixed_size_list(data_type):
        value_type = data_type.value_type
        values = pa.array(matrix.reshape(-1).astype(value_type.to_pandas_dtype()), type=value_type)
        return pa.FixedSizeListArray.from_arrays(values, data_type.list_size)
    return pa.array(matrix[:, 0].astype(data_type.to_pandas_dtype()), type=data_type)


def _is_floating(data_type: pa.DataType) -> bool:
    if pa.types.is_fixed_size_list(data_type):
        return pa.types.is_floating(data_type.value_type)
    return pa.types.is_floating(data_type)


def _interpolate(matrix: NDArray[np.float64], plan: list[PlannedFrame]) -> NDArray[np.float64]:
    """Lay out source rows in plan order, blending inserted rows between their kept neighbors."""
    rows = matrix[[frame.source for frame in plan]].copy()
    for position, frame in enumerate(plan):
        if frame.following is not None:
            rows[position] = (1 - frame.factor) * matrix[frame.source] + frame.factor * matrix[frame.following]
    return rows


def _planned_column(array: pa.Array, plan: list[PlannedFrame]) -> pa.Array:
    """Interpolate floating-point columns and hold every other column from the earlier kept frame."""
    if _is_floating(array.type) and any(frame.following is not None for frame in plan):
        return _from_matrix(_interpolate(_matrix(array), plan), array.type)
    return array.take(pa.array([frame.source for frame in plan], type=pa.int64()))


def _transform_record(transform: ImageTransform | None) -> dict[str, Any] | None:
    if transform is None:
        return None
    return {
        "crop": vars(transform.crop) if transform.crop else None,
        "resize": vars(transform.resize) if transform.resize else None,
    }


class LeRobotExporter:
    """
    Exports episodes of a LeRobot v3.0 dataset, with edits applied, as a new LeRobot v3.0 dataset.

    The output directory must be new or empty. The dataset is written to a temporary sibling
    directory and renamed into place only when complete, so a failed export leaves nothing behind.

    Example:
        >>> exporter = LeRobotExporter("/data/capture/lerobot", "/data/capture-edited", dataset_id="capture--lerobot")
        >>> result = exporter.export_episodes([0], edits_map={0: edits})
    """

    def __init__(self, src_path: str | Path, dst_path: str | Path, dataset_id: str = "") -> None:
        self.src_path = Path(src_path)
        self.dst_path = Path(dst_path)
        self.dataset_id = dataset_id
        self.loader = LeRobotLoader(self.src_path)

    def export_episodes(
        self,
        episode_indices: list[int],
        edits_map: dict[int, EpisodeEditOperations] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ExportResult:
        """
        Export the requested episodes, renumbered from 0, into one derived dataset.

        Args:
            episode_indices: Source episode indices, in output order.
            edits_map: Edit operations by source episode index.
            progress_callback: Optional callback for progress updates.

        Returns:
            ExportResult with the dataset directory and aggregate statistics.
        """
        started = datetime.now(UTC)
        staging: Path | None = None
        try:
            info = self.loader.get_dataset_info()
            if info.codebase_version != _SUPPORTED_VERSION:
                raise LeRobotExportError(
                    f"LeRobot export supports {_SUPPORTED_VERSION} datasets, not {info.codebase_version}"
                )
            if len(set(episode_indices)) != len(episode_indices):
                raise LeRobotExportError("each episode can be exported only once")
            if self.dst_path.exists() and (not self.dst_path.is_dir() or any(self.dst_path.iterdir())):
                raise LeRobotExportError("the output directory must be new or empty")
            episodes = self._episodes(info, episode_indices, edits_map or {})
            sizes = self._video_sizes(info, episodes)
            self.dst_path.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{self.dst_path.name}-", suffix=".partial", dir=self.dst_path.parent)
            )
            self._write(staging, info, episodes, sizes, started, progress_callback)
            if self.dst_path.exists():
                self.dst_path.rmdir()
            staging.rename(self.dst_path)
            staging = None
            removed = sum(
                len({frame for frame in (episode.edits.removed_frames or set()) if 0 <= frame < episode.table.num_rows})
                for episode in episodes
                if episode.edits
            )
            return ExportResult(
                success=True,
                output_files=[str(self.dst_path)],
                stats={
                    "total_episodes": len(episodes),
                    "total_frames": sum(episode.length for episode in episodes),
                    "removed_frames": removed,
                    "duration_ms": (datetime.now(UTC) - started).total_seconds() * 1000,
                },
            )
        except (ExportError, LeRobotLoaderError, ImageTransformError) as error:
            return ExportResult(success=False, output_files=[], error=str(error))
        except Exception as error:
            return ExportResult(success=False, output_files=[], error=f"Unexpected error: {error}")
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def _episodes(
        self, info: LeRobotDatasetInfo, episode_indices: list[int], edits_map: dict[int, EpisodeEditOperations]
    ) -> list[_Episode]:
        unsupported = sorted(name for name, feature in info.features.items() if feature.get("dtype") == "image")
        if unsupported:
            raise LeRobotExportError(f"LeRobot export supports video features only, not image features {unsupported}")
        episodes = []
        offset = 0
        for output_index, source_index in enumerate(episode_indices):
            record = self.loader.episode_record(source_index)
            if record is None:
                raise LeRobotExportError(f"episode {source_index} has no meta/episodes record")
            table = self.loader.load_episode_table(source_index)
            edits = edits_map.get(source_index)
            plan = plan_frames(
                table.num_rows,
                edits.removed_frames if edits else None,
                edits.inserted_frames if edits else None,
            )
            if not plan:
                raise LeRobotExportError(f"episode {source_index} has no frames left after its edits")
            episodes.append(_Episode(source_index, output_index, table, record, edits, plan, offset))
            offset += len(plan)
        return episodes

    def _video_sizes(self, info: LeRobotDatasetInfo, episodes: list[_Episode]) -> dict[str, tuple[int, int]]:
        """Return each camera's output (width, height), identical across the exported episodes."""
        sizes = {}
        for camera, feature in info.features.items():
            if feature.get("dtype") != "video":
                continue
            video_info = feature.get("info", {})
            if video_info.get("video.is_depth_map") or video_info.get("is_depth_map"):
                raise LeRobotExportError(f"depth video {camera} cannot be exported")
            height, width = int(feature["shape"][0]), int(feature["shape"][1])
            outputs = set()
            for episode in episodes:
                transform = episode.transform(camera)
                outputs.add(get_output_dimensions((width, height), transform) if transform else (width, height))
            if len(outputs) != 1:
                raise LeRobotExportError(f"camera {camera} would have different sizes across the exported episodes")
            size = outputs.pop()
            if str(video_info.get("video.pix_fmt", "yuv420p")).startswith("yuv420") and (size[0] % 2 or size[1] % 2):
                raise LeRobotExportError(
                    f"camera {camera} output {size[0]}x{size[1]} must have an even width and height"
                )
            sizes[camera] = size
        return sizes

    def _write(
        self,
        root: Path,
        info: LeRobotDatasetInfo,
        episodes: list[_Episode],
        sizes: dict[str, tuple[int, int]],
        started: datetime,
        progress_callback: ProgressCallback | None,
    ) -> None:
        adjusted = any(episode.edits and episode.edits.trajectory_adjustments for episode in episodes)
        if adjusted and STATE_FEATURE not in info.features:
            raise LeRobotExportError(f"trajectory adjustments need an {STATE_FEATURE} feature")
        tables = [self._episode_table(info, episode, adjusted) for episode in episodes]
        self._write_videos(root, info, episodes, sizes, progress_callback)

        data = pa.concat_tables(tables)
        data_path = root / info.data_path.format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data, data_path)

        features = self._features(info, sizes, adjusted)
        stats_features = [
            name
            for name, feature in features.items()
            if feature.get("dtype") in _STATS_DTYPES and name in data.column_names
        ]
        rows = [
            self._episode_row(info, episode, table, stats_features)
            for episode, table in zip(episodes, tables, strict=True)
        ]
        episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), episodes_path)

        stats: dict[str, Any] = {
            name: _numeric_stats(_matrix(data.column(name).combine_chunks())) for name in stats_features
        }
        for camera in sizes:
            stats[camera] = _image_stats([sample for episode in episodes for sample in episode.image_samples[camera]])
        (root / "meta/stats.json").write_text(json.dumps(stats, indent=4))

        raw = copy.deepcopy(info.raw_info)
        raw.update(
            features=features,
            total_episodes=len(episodes),
            total_frames=data.num_rows,
            splits={"train": f"0:{len(episodes)}"},
        )
        (root / "meta/info.json").write_text(json.dumps(raw, indent=4))
        shutil.copyfile(self.src_path / "meta/tasks.parquet", root / "meta/tasks.parquet")
        (root / PROVENANCE_FILE).write_text(json.dumps(self._provenance(info, episodes, started), indent=2))

    def _episode_table(self, info: LeRobotDatasetInfo, episode: _Episode, adjusted: bool) -> pa.Table:
        """Lay out one episode's output rows with a renumbered timeline and optional derived state."""
        source = episode.table
        plan = episode.plan
        length = len(plan)
        regenerated = {
            "timestamp": np.arange(length) / info.fps,
            "frame_index": np.arange(length),
            "episode_index": np.full(length, episode.output_index),
            "index": episode.offset + np.arange(length),
        }
        columns = {}
        for name in source.column_names:
            column = source.column(name).combine_chunks()
            if name in regenerated:
                columns[name] = pa.array(regenerated[name].astype(column.type.to_pandas_dtype()), type=column.type)
            else:
                columns[name] = _planned_column(column, plan)
        if adjusted:
            state = source.column(STATE_FEATURE).combine_chunks()
            dtype = (state.type.value_type if pa.types.is_fixed_size_list(state.type) else state.type).to_pandas_dtype()
            recorded = _matrix(state)
            adjustments = episode.edits.trajectory_adjustments if episode.edits else None
            changed = apply_trajectory_adjustments(recorded, adjustments) if adjustments else recorded
            planned_recorded = _interpolate(recorded, plan).astype(dtype)
            planned_adjusted = _interpolate(changed, plan).astype(dtype)
            columns[ADJUSTED_STATE] = _from_matrix(planned_adjusted, state.type)
            columns[ADJUSTED_STATE_MASK] = pa.array(np.any(planned_adjusted != planned_recorded, axis=1))
        return pa.table(columns)

    def _write_videos(
        self,
        root: Path,
        info: LeRobotDatasetInfo,
        episodes: list[_Episode],
        sizes: dict[str, tuple[int, int]],
        progress_callback: ProgressCallback | None,
    ) -> None:
        total = max(1, sum(episode.length for episode in episodes) * len(sizes))
        written = 0
        for camera, size in sizes.items():
            video_info = info.features[camera].get("info", {})
            path = root / info.video_path.format(video_key=camera, chunk_index=0, file_index=0)
            output = _VideoOutput(path, video_info, size, info.fps)
            try:
                for episode in episodes:
                    source = self.loader.get_video_path(episode.source_index, camera)
                    window = (
                        float(episode.record.get(f"videos/{camera}/from_timestamp", 0.0)),
                        float(episode.record.get(f"videos/{camera}/to_timestamp", math.inf)),
                    )
                    if source is None:
                        raise LeRobotExportError(f"episode {episode.source_index} has no {camera} video")
                    start = output.frames / info.fps
                    self._write_episode_video(output, source, window, info.fps, episode, camera)
                    episode.windows[camera] = (start, output.frames / info.fps)
                    written += episode.length
                    if progress_callback:
                        progress_callback(
                            ExportProgress(
                                current_episode=episode.source_index,
                                total_episodes=len(episodes),
                                current_frame=episode.length,
                                total_frames=episode.length,
                                percentage=5 + 90 * written / total,
                                status=f"Encoded {camera} for episode {episode.source_index}",
                            )
                        )
            finally:
                output.close()

    @staticmethod
    def _write_episode_video(
        output: _VideoOutput, source: Path, window: tuple[float, float], fps: float, episode: _Episode, camera: str
    ) -> None:
        transform = episode.transform(camera)
        sampled = set(np.linspace(0, episode.length - 1, num=min(episode.length, _IMAGE_SAMPLES)).round().astype(int))
        samples = episode.image_samples.setdefault(camera, [])
        frames = _EpisodeFrames(source, window, fps)
        try:
            for position, frame in enumerate(episode.plan):
                image = frames.get(frame.source, keep_from=frame.source)
                if frame.following is not None:
                    image = interpolate_image(image, frames.get(frame.following, keep_from=frame.source), frame.factor)
                if transform is not None:
                    image = apply_transform(image, transform)
                output.write(image)
                if position in sampled:
                    step = max(1, math.ceil(max(image.shape[:2]) / _IMAGE_STATS_SIZE))
                    samples.append(image[::step, ::step].copy())
            frames.finish(episode.table.num_rows)
        finally:
            frames.close()

    @staticmethod
    def _features(
        info: LeRobotDatasetInfo, sizes: dict[str, tuple[int, int]], adjusted: bool
    ) -> dict[str, dict[str, Any]]:
        features = copy.deepcopy(info.features)
        for camera, (width, height) in sizes.items():
            feature = features[camera]
            feature["shape"] = [height, width, *feature["shape"][2:]]
            feature.setdefault("info", {}).update({"video.height": height, "video.width": width})
        if adjusted:
            state = features[STATE_FEATURE]
            features[ADJUSTED_STATE] = {"dtype": state["dtype"], "shape": state["shape"], "names": state.get("names")}
            features[ADJUSTED_STATE_MASK] = {"dtype": "bool", "shape": [1], "names": None}
        return features

    @staticmethod
    def _episode_row(
        info: LeRobotDatasetInfo, episode: _Episode, table: pa.Table, stats_features: list[str]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "episode_index": episode.output_index,
            "tasks": list(episode.record.get("tasks") or []),
            "length": episode.length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": episode.offset,
            "dataset_to_index": episode.offset + episode.length,
        }
        for camera, (start, end) in episode.windows.items():
            row.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": start,
                    f"videos/{camera}/to_timestamp": end,
                }
            )
        row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        for name in stats_features:
            for stat, value in _numeric_stats(_matrix(table.column(name).combine_chunks())).items():
                row[f"stats/{name}/{stat}"] = value
        for camera, samples in episode.image_samples.items():
            for stat, value in _image_stats(samples).items():
                row[f"stats/{camera}/{stat}"] = value
        return row

    def _provenance(self, info: LeRobotDatasetInfo, episodes: list[_Episode], started: datetime) -> dict[str, Any]:
        records = []
        for episode in episodes:
            edits = episode.edits
            length = episode.table.num_rows
            records.append(
                {
                    "episode_index": episode.output_index,
                    "source_episode_index": episode.source_index,
                    "source_frames": length,
                    "output_frames": episode.length,
                    "frame_sources": [frame.source if frame.following is None else None for frame in episode.plan],
                    "edits": {
                        "removed_frames": sorted(
                            frame for frame in (edits.removed_frames or set()) if 0 <= frame < length
                        )
                        if edits
                        else [],
                        "inserted_frames": [
                            {"after_frame_index": frame.source, "interpolation_factor": frame.factor}
                            for frame in episode.plan
                            if frame.following is not None
                        ],
                        "global_transform": _transform_record(edits.global_transform) if edits else None,
                        "camera_transforms": {
                            camera: _transform_record(transform)
                            for camera, transform in ((edits.camera_transforms or {}) if edits else {}).items()
                        },
                        "trajectory_adjustments": [
                            {
                                "frame_index": adjustment.frame_index,
                                "channel_deltas": adjustment.channel_deltas,
                                "channel_values": adjustment.channel_values,
                            }
                            for adjustment in ((edits.trajectory_adjustments or []) if edits else [])
                        ],
                        "subtasks": remap_subtasks(edits.subtasks or [], output_indices(episode.plan)) if edits else [],
                    },
                }
            )
        return {
            "schema_version": 1,
            "exported_at": started.isoformat(),
            "source": {"dataset_id": self.dataset_id, "codebase_version": info.codebase_version, "fps": info.fps},
            "episodes": records,
        }
