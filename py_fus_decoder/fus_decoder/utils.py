"""Small utility helpers with import-light defaults."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing optional dependency '{module_name}'. Install with: {install_hint}"
        ) from exc


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def maybe_load_yaml_or_json(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        yaml = require_dependency("yaml", 'pip install -e ".[io]"')
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return load_json(path)


def flatten_once(items: Iterable[Sequence[Any]]) -> List[Any]:
    merged: List[Any] = []
    for seq in items:
        merged.extend(seq)
    return merged


def choose_dataset_path_gui(
    title: str = "Select fUS dataset",
    initialdir: str | None = None,
) -> str:
    if platform.system() == "Darwin":
        return choose_dataset_path_macos(title=title, initialdir=initialdir)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(
            "Tkinter GUI is unavailable in this Python environment."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=initialdir,
        filetypes=[
            ("Supported dataset files", "*.mat *.npz"),
            ("MATLAB files", "*.mat"),
            ("NumPy archives", "*.npz"),
            ("All files", "*.*"),
        ],
    )
    if file_path:
        root.destroy()
        return str(Path(file_path).expanduser().resolve())

    dir_path = filedialog.askdirectory(
        title=f"{title} (or choose a dataset directory)",
        initialdir=initialdir,
    )
    root.destroy()
    if dir_path:
        return str(Path(dir_path).expanduser().resolve())

    raise RuntimeError("Dataset selection was cancelled.")


def choose_dataset_paths_gui(
    title: str = "Select fUS datasets",
    initialdir: str | None = None,
) -> List[str]:
    if platform.system() == "Darwin":
        return choose_dataset_paths_macos(title=title, initialdir=initialdir)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(
            "Tkinter GUI is unavailable in this Python environment."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_paths = filedialog.askopenfilenames(
        title=title,
        initialdir=initialdir,
        filetypes=[
            ("MATLAB files", "*.mat"),
            ("All files", "*.*"),
        ],
    )
    if file_paths:
        root.destroy()
        return [str(Path(item).expanduser().resolve()) for item in file_paths]

    dir_path = filedialog.askdirectory(
        title=f"{title} (or choose a dataset directory)",
        initialdir=initialdir,
    )
    root.destroy()
    if dir_path:
        return [str(path.resolve()) for path in sorted(Path(dir_path).expanduser().glob("*.mat"))]

    raise RuntimeError("Dataset selection was cancelled.")


def choose_items_gui(
    title: str,
    prompt: str,
    options: Sequence[str],
    multiple: bool = False,
) -> List[str]:
    if not options:
        raise ValueError("No options are available for selection.")
    if platform.system() == "Darwin":
        return choose_items_macos(title=title, prompt=prompt, options=options, multiple=multiple)

    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    raw = input("Select item number(s), comma-separated: " if multiple else "Select item number: ")
    indexes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not multiple and len(indexes) != 1:
        raise RuntimeError("Please select exactly one item.")
    return [options[index - 1] for index in indexes]


def choose_dataset_path_macos(
    title: str = "Select fUS dataset",
    initialdir: str | None = None,
) -> str:
    initial_path = Path(initialdir).expanduser() if initialdir else Path.cwd()
    if not initial_path.exists():
        initial_path = Path.cwd()

    file_script = _build_macos_choose_script(
        title=title,
        initial_path=initial_path,
        choose_folder=False,
    )
    file_result = subprocess.run(
        ["osascript", "-e", file_script],
        capture_output=True,
        text=True,
    )
    if file_result.returncode == 0 and file_result.stdout.strip():
        return str(Path(file_result.stdout.strip()).expanduser().resolve())

    folder_script = _build_macos_choose_script(
        title=f"{title} (or choose a dataset directory)",
        initial_path=initial_path,
        choose_folder=True,
    )
    folder_result = subprocess.run(
        ["osascript", "-e", folder_script],
        capture_output=True,
        text=True,
    )
    if folder_result.returncode == 0 and folder_result.stdout.strip():
        return str(Path(folder_result.stdout.strip()).expanduser().resolve())

    raise RuntimeError("Dataset selection was cancelled.")


def choose_dataset_paths_macos(
    title: str = "Select fUS datasets",
    initialdir: str | None = None,
) -> List[str]:
    initial_path = Path(initialdir).expanduser() if initialdir else Path.cwd()
    if not initial_path.exists():
        initial_path = Path.cwd()

    file_script = _build_macos_choose_script(
        title=title,
        initial_path=initial_path,
        choose_folder=False,
        multiple=True,
    )
    file_result = subprocess.run(
        ["osascript", "-e", file_script],
        capture_output=True,
        text=True,
    )
    selected_files = _split_osascript_lines(file_result.stdout)
    if file_result.returncode == 0 and selected_files:
        return [str(Path(item).expanduser().resolve()) for item in selected_files]

    folder_script = _build_macos_choose_script(
        title=f"{title} (or choose a dataset directory)",
        initial_path=initial_path,
        choose_folder=True,
    )
    folder_result = subprocess.run(
        ["osascript", "-e", folder_script],
        capture_output=True,
        text=True,
    )
    if folder_result.returncode == 0 and folder_result.stdout.strip():
        folder = Path(folder_result.stdout.strip()).expanduser().resolve()
        return [str(path.resolve()) for path in sorted(folder.glob("*.mat"))]

    raise RuntimeError("Dataset selection was cancelled.")


def choose_items_macos(
    title: str,
    prompt: str,
    options: Sequence[str],
    multiple: bool = False,
) -> List[str]:
    option_list = ", ".join(_applescript_string(item) for item in options)
    multiple_clause = " with multiple selections allowed" if multiple else ""
    script = "\n".join(
        [
            f"set selectedItems to choose from list {{{option_list}}} with title "
            f"{_applescript_string(title)} with prompt {_applescript_string(prompt)}{multiple_clause}",
            'if selectedItems is false then error "Selection cancelled."',
            "set outputText to \"\"",
            "repeat with selectedItem in selectedItems",
            "set outputText to outputText & selectedItem & linefeed",
            "end repeat",
            "outputText",
        ]
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    selected = _split_osascript_lines(result.stdout)
    if result.returncode == 0 and selected:
        return selected
    raise RuntimeError("Selection was cancelled.")


def _build_macos_choose_script(
    title: str,
    initial_path: Path,
    choose_folder: bool,
    multiple: bool = False,
) -> str:
    action = "choose folder" if choose_folder else "choose file"
    type_clause = "" if choose_folder else ' of type {"mat", "npz"}'
    multiple_clause = "" if choose_folder or not multiple else " with multiple selections allowed"
    result_lines = (
        [
            "set outputText to \"\"",
            "repeat with selectedEntry in selectedItem",
            "set outputText to outputText & POSIX path of selectedEntry & linefeed",
            "end repeat",
            "outputText",
        ]
        if multiple
        else ["POSIX path of selectedItem"]
    )
    return "\n".join(
        [
            f'set defaultLocation to POSIX file {_applescript_string(str(initial_path))}',
            (
                f"set selectedItem to {action} with prompt "
                f"{_applescript_string(title)} default location defaultLocation{type_clause}{multiple_clause}"
            ),
            *result_lines,
        ]
    )


def _applescript_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _split_osascript_lines(value: str) -> List[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]
