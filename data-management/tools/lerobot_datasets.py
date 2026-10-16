"""Manage LeRobot datasets in a local data root: list, import, derive learner datasets, and remove.

Datasets live one per directory under the datasets directory, each a complete LeRobot v3 tree with
``meta/info.json``. Derivation selects source episodes from their validation and label sidecars, drops
features from each selected source, and merges the results with ``lerobot-edit-dataset``, staging the work
outside the datasets directory so a partial dataset never appears to readers such as the dataviewer.

Only ``derive`` needs LeRobot; the other commands use the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

INFO_FILE = Path("meta") / "info.json"
LABELS_FILE = Path("meta") / "episode_labels.json"
EXPORT_VALIDATION_FILE = "export-validation.json"
LEDGER_FILE = "ledger.json"
SUCCESS_LABEL = "SUCCESS"
STAGING_DIR_NAME = ".staging"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")

Runner = Callable[[Sequence[str]], None]


class DatasetError(RuntimeError):
    """Raised when a dataset operation cannot proceed safely."""


@dataclass(frozen=True)
class SourceDataset:
    """A LeRobot dataset root and the directory that holds its capture sidecars."""

    root: Path
    sidecar_dir: Path


@dataclass(frozen=True)
class EpisodeSelection:
    """Episodes kept from one source, with the reason when the whole source is excluded."""

    source: SourceDataset
    episodes: tuple[int, ...]
    total_episodes: int
    excluded_reason: str = ""

    @property
    def selected(self) -> bool:
        return bool(self.episodes)


def is_dataset(path: Path) -> bool:
    """Report whether a directory holds a LeRobot dataset."""
    return (path / INFO_FILE).is_file()


def read_info(root: Path) -> dict:
    return json.loads((root / INFO_FILE).read_text(encoding="utf-8"))


def dataset_path(datasets_dir: Path, name: str) -> Path:
    """Return the directory for dataset ``name``, refusing names that escape the datasets directory.

    Raises:
        DatasetError: When the name is not a plain relative path or resolves outside the directory.
    """
    if not _NAME_PATTERN.fullmatch(name) or any(part in {".", ".."} for part in name.split("/")):
        raise DatasetError(f"Invalid dataset name '{name}': use a relative path of letters, digits, '.', '_', and '-'")
    base = datasets_dir.resolve()
    target = (datasets_dir / name).resolve()
    if target == base or not target.is_relative_to(base):
        raise DatasetError(f"Dataset '{name}' resolves outside {base}")
    return target


def find_source(path: Path) -> SourceDataset:
    """Locate the LeRobot dataset in a source directory.

    Accepts a dataset root, or a capture directory whose ``lerobot/`` subdirectory is the dataset and whose
    own files are the capture sidecars.
    """
    if is_dataset(path):
        return SourceDataset(root=path, sidecar_dir=path)
    if is_dataset(path / "lerobot"):
        return SourceDataset(root=path / "lerobot", sidecar_dir=path)
    raise DatasetError(f"No LeRobot dataset at {path} (expected meta/info.json or lerobot/meta/info.json)")


def select_episodes(source: SourceDataset, *, include_unvalidated: bool = False) -> EpisodeSelection:
    """Keep the episodes a source's sidecars show as valid and successful.

    * ``export-validation.json`` must report ``status: pass`` for the whole dataset.
    * ``ledger.json`` keeps only episodes whose ``status`` is ``passed``.
    * ``meta/episode_labels.json`` keeps only episodes labeled ``SUCCESS``.

    A source without a validation sidecar is excluded unless ``include_unvalidated`` is set.
    """
    total = int(read_info(source.root).get("total_episodes", 0))
    kept = set(range(total))

    def excluded(reason: str) -> EpisodeSelection:
        return EpisodeSelection(source=source, episodes=(), total_episodes=total, excluded_reason=reason)

    validation_path = source.sidecar_dir / EXPORT_VALIDATION_FILE
    ledger_path = source.sidecar_dir / LEDGER_FILE
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "pass":
            failed = ", ".join(validation.get("failed_groups") or []) or "unknown"
            return excluded(f"export validation {validation.get('status', 'missing')} (failed: {failed})")
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        kept &= {int(episode["index"]) for episode in ledger.get("episodes", []) if episode.get("status") == "passed"}
    if not validation_path.is_file() and not ledger_path.is_file() and not include_unvalidated:
        return excluded(f"no {EXPORT_VALIDATION_FILE} or {LEDGER_FILE}")

    labels_path = source.root / LABELS_FILE
    if labels_path.is_file():
        labels = json.loads(labels_path.read_text(encoding="utf-8")).get("episodes", {})
        kept &= {int(index) for index, names in labels.items() if SUCCESS_LABEL in names}

    if not kept:
        return excluded("no passing, successful episodes")
    return EpisodeSelection(source=source, episodes=tuple(sorted(kept)), total_episodes=total)


def remove_feature_command(
    executable: Sequence[str], *, root: Path, new_root: Path, repo_id: str, new_repo_id: str, features: Sequence[str]
) -> list[str]:
    return [
        *executable,
        "--repo_id",
        repo_id,
        "--root",
        str(root),
        "--new_repo_id",
        new_repo_id,
        "--new_root",
        str(new_root),
        "--operation.type",
        "remove_feature",
        "--operation.feature_names",
        json.dumps(list(features)),
    ]


def delete_episodes_command(
    executable: Sequence[str], *, root: Path, repo_id: str, episodes: Sequence[int]
) -> list[str]:
    return [
        *executable,
        "--repo_id",
        repo_id,
        "--root",
        str(root),
        "--operation.type",
        "delete_episodes",
        "--operation.episode_indices",
        json.dumps(list(episodes)),
    ]


def merge_command(
    executable: Sequence[str], *, roots: Sequence[Path], repo_ids: Sequence[str], new_root: Path, new_repo_id: str
) -> list[str]:
    return [
        *executable,
        "--new_repo_id",
        new_repo_id,
        "--new_root",
        str(new_root),
        "--operation.type",
        "merge",
        "--operation.repo_ids",
        json.dumps(list(repo_ids)),
        "--operation.roots",
        json.dumps([str(root) for root in roots]),
    ]


def _run_checked(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def derive(
    datasets_dir: Path,
    name: str,
    sources: Sequence[Path],
    *,
    drop_features: Sequence[str] = (),
    include_unvalidated: bool = False,
    executable: Sequence[str] = ("lerobot-edit-dataset",),
    run: Runner = _run_checked,
    log: Callable[[str], None] = print,
) -> Path:
    """Build dataset ``name`` from the passing, successful episodes of ``sources`` without ``drop_features``.

    Returns:
        The new dataset directory.

    Raises:
        DatasetError: When the target exists or no source contributes an episode.
    """
    target = dataset_path(datasets_dir, name)
    if target.exists():
        raise DatasetError(f"Dataset '{name}' already exists at {target}; remove it first")

    selections = [select_episodes(find_source(path), include_unvalidated=include_unvalidated) for path in sources]
    for selection in selections:
        if selection.selected:
            log(f"Using {selection.source.root}: episodes {list(selection.episodes)} of {selection.total_episodes}")
        else:
            log(f"Skipping {selection.source.root}: {selection.excluded_reason}")
    chosen = [selection for selection in selections if selection.selected]
    if not chosen:
        raise DatasetError("No source has a passing, successful episode")

    # Staging sits beside the datasets directory on the same filesystem, so the finished dataset moves in
    # with one rename and readers never see a partial tree.
    staging = datasets_dir.resolve().parent / STAGING_DIR_NAME / f"{name.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        parts: list[Path] = []
        part_ids: list[str] = []
        for index, selection in enumerate(chosen):
            part, part_id = staging / f"part-{index}", f"local/part-{index}"
            source_id = f"local/source-{index}"
            features = read_info(selection.source.root).get("features", {})
            present = [feature for feature in drop_features if feature in features]
            if present:
                run(
                    remove_feature_command(
                        executable,
                        root=selection.source.root,
                        new_root=part,
                        repo_id=source_id,
                        new_repo_id=part_id,
                        features=present,
                    )
                )
            else:
                shutil.copytree(selection.source.root, part)
            dropped = sorted(set(range(selection.total_episodes)) - set(selection.episodes))
            if dropped:
                run(delete_episodes_command(executable, root=part, repo_id=part_id, episodes=dropped))
            parts.append(part)
            part_ids.append(part_id)

        result = parts[0]
        if len(parts) > 1:
            result = staging / "merged"
            run(merge_command(executable, roots=parts, repo_ids=part_ids, new_root=result, new_repo_id=f"local/{name}"))
        if not is_dataset(result):
            raise DatasetError(f"lerobot-edit-dataset did not produce a dataset at {result}")
        target.parent.mkdir(parents=True, exist_ok=True)
        result.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"Created {target}")
    return target


def import_dataset(datasets_dir: Path, name: str, source: Path, *, log: Callable[[str], None] = print) -> Path:
    """Copy a LeRobot dataset into the datasets directory as ``name``."""
    target = dataset_path(datasets_dir, name)
    if target.exists():
        raise DatasetError(f"Dataset '{name}' already exists at {target}; remove it first")
    root = find_source(source).root
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, target)
    log(f"Imported {root} as {target}")
    return target


def list_datasets(datasets_dir: Path) -> list[dict[str, object]]:
    """Describe every dataset under the datasets directory, nested names included."""
    rows: list[dict[str, object]] = []
    if not datasets_dir.is_dir():
        return rows
    for info_path in sorted(datasets_dir.rglob(str(INFO_FILE))):
        root = info_path.parent.parent
        if any(part.startswith(".") for part in root.relative_to(datasets_dir).parts):
            continue
        info = read_info(root)
        cameras = [key for key, value in info.get("features", {}).items() if value.get("dtype") in ("video", "image")]
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        rows.append(
            {
                "name": root.relative_to(datasets_dir).as_posix(),
                "episodes": info.get("total_episodes", 0),
                "frames": info.get("total_frames", 0),
                "fps": info.get("fps", 0),
                "cameras": len(cameras),
                "size_mb": round(size / 1_000_000, 1),
            }
        )
    return rows


def remove_dataset(
    datasets_dir: Path,
    name: str,
    *,
    assume_yes: bool = False,
    confirm: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
) -> bool:
    """Delete dataset ``name`` after confirmation.

    Returns:
        True when the dataset was removed, False when the confirmation was declined.

    Raises:
        DatasetError: When the name escapes the datasets directory or is not a dataset.
    """
    target = dataset_path(datasets_dir, name)
    if not is_dataset(target):
        raise DatasetError(f"No dataset named '{name}' at {target}")
    if not assume_yes and confirm(f"Remove {target}? [y/N] ").strip().lower() not in {"y", "yes"}:
        log("Left unchanged")
        return False
    shutil.rmtree(target)
    log(f"Removed {target}")
    return True


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets-dir", type=Path, required=True, help="Directory that holds one dataset per folder")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List datasets")

    import_parser = commands.add_parser("import", help="Copy a LeRobot dataset in")
    import_parser.add_argument("source", type=Path, help="Dataset root, or a capture directory with lerobot/")
    import_parser.add_argument("name", help="Dataset name under the datasets directory")

    derive_parser = commands.add_parser("derive", help="Merge passing episodes and drop features into a new dataset")
    derive_parser.add_argument("name", help="New dataset name under the datasets directory")
    derive_parser.add_argument(
        "--source", type=Path, action="append", required=True, help="Source dataset (repeatable)"
    )
    derive_parser.add_argument("--drop-feature", action="append", default=[], help="Feature to remove (repeatable)")
    derive_parser.add_argument(
        "--include-unvalidated", action="store_true", help="Use sources that have no validation sidecar"
    )

    remove_parser = commands.add_parser("remove", help="Delete a dataset after confirmation")
    remove_parser.add_argument("name", help="Dataset name under the datasets directory")
    remove_parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    datasets_dir: Path = args.datasets_dir
    try:
        if args.command == "list":
            rows = list_datasets(datasets_dir)
            if not rows:
                print(f"No datasets in {datasets_dir}")
            for row in rows:
                print(
                    f"{row['name']}\tepisodes={row['episodes']}\tframes={row['frames']}\tfps={row['fps']}"
                    f"\tcameras={row['cameras']}\tsize={row['size_mb']}MB"
                )
        elif args.command == "import":
            import_dataset(datasets_dir, args.name, args.source)
        elif args.command == "derive":
            if shutil.which("lerobot-edit-dataset") is None:
                raise DatasetError("lerobot-edit-dataset is not on PATH; run derive from the LeRobot environment")
            derive(
                datasets_dir,
                args.name,
                args.source,
                drop_features=args.drop_feature,
                include_unvalidated=args.include_unvalidated,
            )
        elif args.command == "remove":
            remove_dataset(datasets_dir, args.name, assume_yes=args.yes)
    except (DatasetError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
