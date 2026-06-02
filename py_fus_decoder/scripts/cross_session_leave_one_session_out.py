#!/usr/bin/env python3
"""Cross-session leave-one-session-out evaluation for fUS decoding."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_MANIFEST = ROOT / "configs" / "cross_session_saccade8_real.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "cross_session_saccade8_real"
DEFAULT_MPLCONFIGDIR = ROOT / "output" / ".mplconfig"
PROJECT_ROOT = ROOT.parent
PROJECT_RECORD = ROOT / "dataset" / "project_record.json"
SESSION_FILE_PATTERN = re.compile(r"rt_fUS_data_S(?P<session>\d+)_R(?P<run>\d+)\.mat$")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

from fus_decoder.config import DatasetConfig, ModelConfig  # noqa: E402
from fus_decoder.data import load_dataset  # noqa: E402
from fus_decoder.models import build_model  # noqa: E402
from fus_decoder.utils import (  # noqa: E402
    choose_dataset_paths_gui,
    choose_items_gui,
    load_json,
    maybe_load_yaml_or_json,
    require_dependency,
    save_json,
)
from group_sessions_by_condition import group_sessions  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-session leave-one-session-out evaluation."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to session manifest JSON/YAML. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save results. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model-names",
        nargs="*",
        default=None,
        help="Optional subset of model names to run.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run evaluation without generating matplotlib figures.",
    )
    parser.add_argument(
        "--gui-select-datasets",
        action="store_true",
        default=True,
        help="Open GUI dialogs to select datasets, group, and LOSO sessions.",
    )
    parser.add_argument(
        "--no-gui-select-datasets",
        dest="gui_select_datasets",
        action="store_false",
        help="Use sessions from manifest without GUI dialogs.",
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=3,
        help="Minimum sessions required for LOSO. Default: 3.",
    )
    return parser


def resolve_config_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        PROJECT_ROOT / path,
        ROOT / path,
        manifest_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def infer_session_run_from_path(path: Path) -> Optional[Tuple[int, int]]:
    match = SESSION_FILE_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group("session")), int(match.group("run"))


def load_project_records() -> List[Dict[str, Any]]:
    if not PROJECT_RECORD.exists():
        return []
    return load_json(PROJECT_RECORD)


def find_session_record(
    records: List[Dict[str, Any]],
    session_id: int,
    run_id: int,
) -> Optional[Dict[str, Any]]:
    for item in records:
        if int(item.get("Session", -1)) == session_id and int(item.get("Run", -1)) == run_id:
            return item
    return None


def build_session_entries_from_paths(paths: List[str]) -> List[Dict[str, Any]]:
    records = load_project_records()
    entries = []
    for raw_path in sorted({str(Path(item).expanduser().resolve()) for item in paths}):
        path = Path(raw_path)
        session_run = infer_session_run_from_path(path)
        entry: Dict[str, Any] = {"path": str(path)}
        if session_run is not None:
            session_id, run_id = session_run
            record = find_session_record(records, session_id, run_id)
            if record:
                entry.update(record)
            else:
                entry.update({"Session": session_id, "Run": run_id})
        entries.append(entry)
    return entries


def session_label(session: Dict[str, Any]) -> str:
    return (
        f"{session['session_id']}_{session['run_id']} | "
        f"{session.get('date', session.get('Date', 'unknown date'))} | "
        f"Monkey={session.get('monkey', session.get('Monkey', '?'))}, "
        f"Slot={session.get('slot', session.get('Slot', '?'))}, "
        f"Task={session.get('task', session.get('Task', '?'))}, "
        f"nTargets={session.get('n_targets', session.get('nTargets', '?'))}"
    )


def group_label(group: Dict[str, Any]) -> str:
    return (
        f"{group['merge_key']} | "
        f"{group['n_sessions']} sessions: {', '.join(group['session_members'])}"
    )


def select_group_and_sessions_gui(
    groups: List[Dict[str, Any]],
    min_sessions: int,
) -> List[Dict[str, Any]]:
    eligible = [group for group in groups if group["n_sessions"] >= min_sessions]
    if not eligible:
        return []

    if len(eligible) == 1:
        selected_group = eligible[0]
    else:
        labels = [group_label(group) for group in eligible]
        selected_label = choose_items_gui(
            title="Select cross-session group",
            prompt="Choose one group with identical Monkey, Slot, Task, and nTargets.",
            options=labels,
            multiple=False,
        )[0]
        selected_group = eligible[labels.index(selected_label)]

    sessions = selected_group["sessions"]
    if len(sessions) == min_sessions:
        selected_sessions = sessions
    else:
        labels = [session_label(session) for session in sessions]
        selected_labels = choose_items_gui(
            title="Select LOSO sessions",
            prompt=f"Choose at least {min_sessions} sessions for leave-one-session-out.",
            options=labels,
            multiple=True,
        )
        if len(selected_labels) < min_sessions:
            raise RuntimeError(f"Please choose at least {min_sessions} sessions for LOSO.")
        selected_sessions = [sessions[labels.index(label)] for label in selected_labels]

    return [
        {
            **selected_group,
            "sessions": selected_sessions,
            "session_members": [
                f"{item['session_id']}_{item['run_id']}" for item in selected_sessions
            ],
            "n_sessions": len(selected_sessions),
        }
    ]


def independent_session_zscore(samples: Any) -> Any:
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    samples = np.asarray(samples, dtype=np.float32)
    flat = samples.reshape(samples.shape[0], -1)
    mu = flat.mean(axis=0, keepdims=True)
    sigma = flat.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    flat = (flat - mu) / sigma
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.reshape(samples.shape)


def circular_angle_error_deg(true_angles: Any, pred_angles: Any) -> Any:
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    diff = np.abs(np.asarray(true_angles) - np.asarray(pred_angles))
    return np.minimum(diff, 360.0 - diff)


def label_angle_map_from_sessions(sessions: List[Any], labels: List[int]) -> Dict[int, float]:
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    points_by_label: Dict[int, List[float]] = {int(label): [] for label in labels}

    for session in sessions:
        metadata = getattr(session, "metadata", {})
        if "target_pos_x" not in metadata or "target_pos_y" not in metadata:
            continue
        xs = np.asarray(metadata["target_pos_x"])
        ys = np.asarray(metadata["target_pos_y"])
        session_labels = np.asarray(session.labels)
        for label in labels:
            mask = session_labels == label
            if np.any(mask):
                angles = np.degrees(np.arctan2(ys[mask], xs[mask])) % 360.0
                points_by_label[int(label)].extend(angles.tolist())

    fallback = equally_spaced_label_angle_map(labels)
    angle_map = {}
    for label in labels:
        values = points_by_label[int(label)]
        if values:
            radians = np.radians(values)
            mean_angle = np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0
            angle_map[int(label)] = float(mean_angle)
        else:
            angle_map[int(label)] = fallback[int(label)]
    return angle_map


def equally_spaced_label_angle_map(labels: List[int]) -> Dict[int, float]:
    labels_sorted = sorted(int(label) for label in labels)
    step = 360.0 / max(1, len(labels_sorted))
    return {label: float(idx * step) for idx, label in enumerate(labels_sorted)}


def true_angles_for_session(session: Any, y_true: Any, angle_map: Dict[int, float]) -> Any:
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    metadata = getattr(session, "metadata", {})
    if "target_pos_x" in metadata and "target_pos_y" in metadata:
        xs = np.asarray(metadata["target_pos_x"])
        ys = np.asarray(metadata["target_pos_y"])
        if len(xs) == len(y_true) and len(ys) == len(y_true):
            return np.degrees(np.arctan2(ys, xs)) % 360.0
    return labels_to_angles(y_true, angle_map)


def labels_to_angles(labels: Any, angle_map: Dict[int, float]) -> Any:
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    return np.asarray([angle_map[int(label)] for label in labels], dtype=np.float64)


def load_session_dataset(session_entry: Dict[str, Any], dataset_defaults: Dict[str, Any]) -> Any:
    extra = dict(dataset_defaults.get("extra", {}))
    dataset_cfg = DatasetConfig(
        name=session_entry.get("name", f"{session_entry['session_id']}_{session_entry['run_id']}"),
        path=session_entry["path"],
        loader=dataset_defaults.get("loader", "mat"),
        group_key=dataset_defaults.get("group_key"),
        species=session_entry.get("species", dataset_defaults.get("species")),
        task=session_entry.get("task", dataset_defaults.get("task")),
        shape_hint=dataset_defaults.get("shape_hint"),
        extra=extra,
    )
    dataset = load_dataset(dataset_cfg)
    dataset.samples = independent_session_zscore(dataset.samples)
    dataset.metadata = dict(dataset.metadata)
    dataset.metadata["session_id"] = session_entry["session_id"]
    dataset.metadata["run_id"] = session_entry["run_id"]
    return dataset


def run_cross_session_group(
    group: Dict[str, Any],
    dataset_defaults: Dict[str, Any],
    model_payloads: List[Dict[str, Any]],
    output_dir: Path,
    save_plots: bool = True,
) -> Dict[str, Any]:
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    metrics = require_dependency("sklearn.metrics", 'pip install -e ".[classical]"')

    sessions = [load_session_dataset(item, dataset_defaults) for item in group["sessions"]]
    models = [ModelConfig(**payload) for payload in model_payloads if payload.get("enabled", True)]

    group_result: Dict[str, Any] = {
        "group_key": group["merge_key"],
        "monkey": group["monkey"],
        "slot": group["slot"],
        "task": group["task"],
        "n_targets": group["n_targets"],
        "n_sessions": group["n_sessions"],
        "session_members": group["session_members"],
        "models": [],
    }

    for model_cfg in models:
        fold_results = []
        y_true_all = []
        y_pred_all = []
        labels_union = sorted({int(label) for session in sessions for label in np.unique(session.labels)})

        for heldout_idx, test_session in enumerate(sessions):
            train_sessions = [s for idx, s in enumerate(sessions) if idx != heldout_idx]
            X_train = np.concatenate([s.samples for s in train_sessions], axis=0)
            y_train = np.concatenate([np.asarray(s.labels) for s in train_sessions], axis=0)
            X_test = np.asarray(test_session.samples)
            y_test = np.asarray(test_session.labels)
            angle_map = label_angle_map_from_sessions(train_sessions, labels_union)

            model_params = dict(model_cfg.params)
            if model_cfg.family in {"pca_lda", "cpca_lda"}:
                model_params["apply_standard_scaler"] = False
            model = build_model(
                ModelConfig(
                    name=model_cfg.name,
                    family=model_cfg.family,
                    enabled=True,
                    params=model_params,
                )
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(metrics.accuracy_score(y_test, y_pred))
            true_angles = true_angles_for_session(test_session, y_test, angle_map)
            pred_angles = labels_to_angles(y_pred, angle_map)
            angular_errors = circular_angle_error_deg(true_angles, pred_angles)
            mean_angular_error = float(np.mean(angular_errors))
            cm = metrics.confusion_matrix(y_test, y_pred, labels=labels_union)
            report = metrics.classification_report(
                y_test,
                y_pred,
                labels=labels_union,
                output_dict=True,
                zero_division=0,
            )
            fold_results.append(
                {
                    "heldout_session": f"{group['sessions'][heldout_idx]['session_id']}_{group['sessions'][heldout_idx]['run_id']}",
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "accuracy": acc,
                    "mean_angular_error_deg": mean_angular_error,
                    "confusion_matrix": cm.tolist(),
                    "classification_report": report,
                }
            )
            y_true_all.append(np.asarray(y_test))
            y_pred_all.append(np.asarray(y_pred))

        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)
        mean_acc = float(metrics.accuracy_score(y_true_cat, y_pred_cat))
        fold_sizes = [fold["n_test"] for fold in fold_results]
        mean_angular_error = float(
            np.average(
                [fold["mean_angular_error_deg"] for fold in fold_results],
                weights=fold_sizes,
            )
        )
        cm_all = metrics.confusion_matrix(y_true_cat, y_pred_cat, labels=labels_union)
        report_all = metrics.classification_report(
            y_true_cat,
            y_pred_cat,
            labels=labels_union,
            output_dict=True,
            zero_division=0,
        )

        model_result = {
            "name": model_cfg.name,
            "family": model_cfg.family,
            "params": model_cfg.params,
            "folds": fold_results,
            "fold_accuracies": [fold["accuracy"] for fold in fold_results],
            "fold_mean_angular_error_deg": [
                fold["mean_angular_error_deg"] for fold in fold_results
            ],
            "mean_accuracy": mean_acc,
            "mean_angular_error_deg": mean_angular_error,
            "confusion_matrix": cm_all.tolist(),
            "classification_report": report_all,
            "labels": labels_union,
        }
        group_result["models"].append(model_result)

        if save_plots:
            plot_confusion_matrix(
                cm_all,
                labels_union,
                output_dir / f"{group['merge_key']}__{model_cfg.name}__confusion.png",
                title=f"{group['merge_key']} | {model_cfg.name}",
            )

    if save_plots:
        plot_accuracy_bar(
            group_result["models"],
            output_dir / f"{group['merge_key']}__accuracy_bar.png",
            title=f"Cross-session accuracy | {group['merge_key']}",
        )
    return group_result


def plot_accuracy_bar(models: List[Dict[str, Any]], output_path: Path, title: str) -> None:
    plt = require_dependency("matplotlib.pyplot", 'pip install matplotlib')
    names = [item["name"] for item in models]
    values = [item["mean_accuracy"] for item in models]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: Any,
    labels: List[int],
    output_path: Path,
    title: str,
) -> None:
    plt = require_dependency("matplotlib.pyplot", 'pip install matplotlib')
    np = require_dependency("numpy", 'pip install -e ".[full]"')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i][j])), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_summary(results: Dict[str, Any], output_dir: Path) -> None:
    print("")
    print("Cross-session LOSO summary")
    print(f"- Eligible groups: {results['n_eligible_groups']}")
    if not results["groups"]:
        print("- No group had at least two existing session files.")
        if results["excluded_sessions"]:
            print("- Excluded sessions:")
            for item in results["excluded_sessions"]:
                print(
                    f"  {item.get('session_id')}_{item.get('run_id')}: "
                    f"{item.get('reason')} path={item.get('path')}"
                )
        return

    for group in results["groups"]:
        print(f"- Group: {group['group_key']}")
        print(f"  Sessions: {', '.join(group['session_members'])}")
        for model in group["models"]:
            fold_text = ", ".join(
                (
                    f"{fold['heldout_session']}="
                    f"acc:{fold['accuracy']:.4f}, "
                    f"mae:{fold['mean_angular_error_deg']:.2f}deg"
                )
                for fold in model["folds"]
            )
            print(
                f"  {model['name']} [{model['family']}]: "
                f"mean_acc={model['mean_accuracy']:.4f}, "
                f"mean_ang_err={model['mean_angular_error_deg']:.2f}deg; "
                f"folds: {fold_text}"
            )

    print(f"- Results: {output_dir / 'cross_session_summary.json'}")
    print(f"- Figures: {output_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = maybe_load_yaml_or_json(manifest_path)
    sessions = payload["sessions"]
    dataset_defaults = payload.get("dataset_defaults", {})
    models = payload["models"]
    dataset_root = None
    if payload.get("dataset_root"):
        dataset_root = resolve_config_path(str(payload["dataset_root"]), manifest_path)
    if args.gui_select_datasets:
        initialdir = str(dataset_root or (ROOT / "dataset"))
        selected_paths = choose_dataset_paths_gui(
            title="Select 3 or more fUS session .mat files, or choose a dataset folder",
            initialdir=initialdir,
        )
        if len(selected_paths) < args.min_sessions:
            raise RuntimeError(f"Please select at least {args.min_sessions} .mat session files.")
        sessions = build_session_entries_from_paths(selected_paths)
        dataset_root = None
        print("Selected dataset files:")
        for path in selected_paths:
            print(f"- {path}")
    if args.model_names:
        wanted = set(args.model_names)
        models = [model for model in models if model.get("name") in wanted]
        if not models:
            raise ValueError(f"No models matched --model-names={sorted(wanted)}")

    grouping = group_sessions(sessions, dataset_root=dataset_root)
    groups_with_existing_data = []
    excluded_missing_paths = []
    for group in grouping["groups"]:
        valid_sessions = [session for session in group["sessions"] if session.get("path") and session.get("path_exists", False)]
        if len(valid_sessions) >= args.min_sessions:
            groups_with_existing_data.append(
                {
                    **group,
                    "sessions": valid_sessions,
                    "session_members": [
                        f"{item['session_id']}_{item['run_id']}" for item in valid_sessions
                    ],
                    "n_sessions": len(valid_sessions),
                }
            )
        else:
            excluded_missing_paths.extend(
                [
                    {
                        **session,
                        "reason": "Session file is missing or fewer than two existing sessions remain in this group.",
                    }
                    for session in group["sessions"]
                    if not session.get("path_exists", False) or len(valid_sessions) < args.min_sessions
                ]
            )

    if args.gui_select_datasets:
        groups_with_existing_data = select_group_and_sessions_gui(
            groups_with_existing_data,
            min_sessions=args.min_sessions,
        )

    results = {
        "rule": grouping["rule"],
        "n_input_sessions": grouping["n_input_sessions"],
        "normalization": "Per-session independent z-score normalization is applied before cross-session train/test splitting.",
        "anti_leakage": "Each held-out session is never used in model fitting. PCA/cPCA/LDA parameters are fit on training sessions only. Test-session normalization uses only test-session statistics.",
        "n_eligible_groups": len(groups_with_existing_data),
        "groups": [],
        "excluded_sessions": grouping["excluded_sessions"] + excluded_missing_paths,
    }

    for group in groups_with_existing_data:
        group_result = run_cross_session_group(
            group,
            dataset_defaults,
            models,
            output_dir,
            save_plots=not args.skip_plots,
        )
        results["groups"].append(group_result)

    save_json(output_dir / "cross_session_summary.json", results)
    print(f"Saved cross-session results to {output_dir / 'cross_session_summary.json'}")
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
