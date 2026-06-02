#!/usr/bin/env python3
"""Group fUS sessions by merge-safe experimental conditions.

Rule:
- For cross-session generalization experiments, sessions are grouped only
  when monkey, slot, task, and nTargets are identical.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fus_decoder.utils import maybe_load_yaml_or_json, save_json  # noqa: E402


FILENAME_PATTERN = re.compile(r"rt_fUS_data_S(?P<session>\d+)_R(?P<run>\d+)\.mat$")
DEFAULT_MANIFEST = ROOT / "dataset" / "project_record.json"
DEFAULT_OUTPUT = ROOT / "output" / "project_record_groups.json"
DEFAULT_DATASET_ROOT = ROOT / "dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group sessions by merge-safe experimental conditions."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to JSON/YAML session manifest. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to output JSON report. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Optional dataset root used to infer paths like rt_fUS_data_S*_R*.mat.",
    )
    return parser


def normalize_session_entry(
    entry: Dict[str, Any],
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    normalized = canonicalize_session_metadata(entry)
    session_token = normalized.get("session_id", normalized.get("Session"))
    run_token = normalized.get("run_id", normalized.get("Run"))
    normalized["session_id"] = format_session_id(session_token)
    normalized["run_id"] = format_run_id(run_token)

    path_value = normalized.get("path")
    if path_value:
        path = Path(str(path_value)).expanduser().resolve()
    elif dataset_root is not None and session_token is not None and run_token is not None:
        path = infer_dataset_path(dataset_root, session_token, run_token)
        normalized["path_inferred"] = True
    else:
        path = None

    if path is not None:
        normalized["path"] = str(path)
        normalized["path_exists"] = path.exists()
        normalized["session_id"] = infer_session_id(path)
        normalized["run_id"] = infer_run_id(path)
    else:
        normalized["path"] = None
        normalized["path_exists"] = False

    normalized.setdefault("merge_key", build_merge_key(normalized))
    normalized.setdefault("can_merge", can_merge_entry(normalized))
    return normalized


def infer_dataset_path(dataset_root: Path, session_id: Any, run_id: Any) -> Path:
    session_num = int(str(session_id).replace("S", "").strip())
    run_num = int(str(run_id).replace("R", "").strip())
    return dataset_root / f"rt_fUS_data_S{session_num}_R{run_num}.mat"


def format_session_id(value: Any) -> str:
    if value is None:
        return "unknown"
    token = str(value).strip()
    if token.lower().startswith("s"):
        suffix = token[1:]
    else:
        suffix = token
    return f"S{suffix}"


def format_run_id(value: Any) -> str:
    if value is None:
        return "unknown"
    token = str(value).strip()
    if token.lower().startswith("r"):
        suffix = token[1:]
    else:
        suffix = token
    return f"R{suffix}"


def infer_session_id(path: Path) -> str:
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        return path.stem
    return f"S{int(match.group('session'))}"


def infer_run_id(path: Path) -> str:
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        return "unknown"
    return f"R{int(match.group('run'))}"


def can_merge_entry(entry: Dict[str, Any]) -> bool:
    monkey = str(entry.get("monkey", "")).strip()
    slot = str(entry.get("slot", "")).strip()
    task = str(entry.get("task", "")).strip()
    n_targets = str(entry.get("n_targets", "")).strip()
    return bool(monkey and slot and task and n_targets)


def build_merge_key(entry: Dict[str, Any]) -> str:
    monkey = str(entry.get("monkey", "")).strip().lower()
    slot = str(entry.get("slot", "")).strip().lower()
    task = str(entry.get("task", "")).strip().lower()
    n_targets = str(entry.get("n_targets", "")).strip().lower()
    if not monkey or not slot or not task or not n_targets:
        return "UNMERGEABLE"
    return f"monkey={monkey}__slot={slot}__task={task}__n_targets={n_targets}"


def group_sessions(
    entries: List[Dict[str, Any]],
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    excluded: List[Dict[str, Any]] = []

    for raw_entry in entries:
        entry = normalize_session_entry(raw_entry, dataset_root=dataset_root)
        if not entry["can_merge"]:
            excluded.append(
                {
                    **entry,
                    "reason": "Missing required merge condition: monkey, slot, task and/or n_targets.",
                }
            )
            continue
        groups.setdefault(entry["merge_key"], []).append(entry)

    group_reports = []
    for merge_key, members in sorted(groups.items()):
        member_ids = [f"{item['session_id']}_{item['run_id']}" for item in members]
        group_reports.append(
            {
                "merge_key": merge_key,
                "monkey": members[0]["monkey"],
                "slot": members[0]["slot"],
                "task": members[0]["task"],
                "n_targets": members[0]["n_targets"],
                "n_sessions": len(members),
                "session_members": member_ids,
                "sessions": members,
                "recommend_joint_training": len(members) >= 2,
            }
        )

    return {
        "rule": "Only sessions with identical monkey, slot, task, and n_targets are grouped for cross-session leave-one-session-out evaluation.",
        "n_input_sessions": len(entries),
        "n_groups": len(group_reports),
        "groups": group_reports,
        "excluded_sessions": excluded,
    }


def canonicalize_session_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(entry)
    alias_map = {
        "Session": "session_id",
        "Run": "run_id",
        "Monkey": "monkey",
        "Slot": "slot",
        "Task": "task",
        "nTargets": "n_targets",
        "nTrials": "n_trials",
        "Date": "date",
        "Notes": "notes",
    }
    for src, dst in alias_map.items():
        if src in normalized and dst not in normalized:
            normalized[dst] = normalized[src]
    return normalized


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None

    payload = maybe_load_yaml_or_json(manifest_path)
    if isinstance(payload, dict) and "sessions" in payload:
        sessions = payload["sessions"]
        if dataset_root is None and payload.get("dataset_root"):
            dataset_root = Path(str(payload["dataset_root"])).expanduser().resolve()
    elif isinstance(payload, list):
        sessions = payload
    else:
        raise ValueError("Manifest must be a list of sessions or a dict with a 'sessions' field.")

    report = group_sessions(sessions, dataset_root=dataset_root)
    save_json(output_path, report)

    print(f"Grouped {report['n_input_sessions']} sessions into {report['n_groups']} merge-safe groups.")
    for group in report["groups"]:
        print(
            f"- {group['merge_key']}: {group['n_sessions']} session(s), "
            f"joint_training={'yes' if group['recommend_joint_training'] else 'no'}"
        )
    if report["excluded_sessions"]:
        print(f"Excluded {len(report['excluded_sessions'])} session(s) with incomplete conditions.")


if __name__ == "__main__":
    main()
