from pathlib import Path
from typing import Iterable, Optional, Tuple


def project_root_from_cwd() -> Path:
    """Return the data/results root inferred from the current working directory.

    This project currently keeps analysis code in ``irt and k factors/`` and
    generated model results in the parent folder's ``results/`` directory.
    """
    data_root, _ = resolve_roots(Path.cwd())
    return data_root


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_first(base: Path, names: Iterable[str], required: bool = True) -> Optional[Path]:
    """Find the first existing file among candidate names under base."""
    for name in names:
        p = base / name
        if p.exists():
            return p
    if required:
        tried = "\n".join(str(base / n) for n in names)
        raise FileNotFoundError(f"Could not find any candidate file. Tried:\n{tried}")
    return None


def resolve_roots(start: Optional[Path] = None) -> Tuple[Path, Path]:
    """Infer ``(data_root, code_root)`` for both supported folder layouts.

    Supported layouts:
    - Current repo layout:
      final project/
        results/
        irt and k factors/src/disaster_irt/
    - Self-contained layout:
      irt and k factors/
        results/
        src/disaster_irt/
    """
    root = Path(start).resolve() if start is not None else Path.cwd().resolve()

    if (root / "src" / "disaster_irt").exists():
        code_root = root
        data_root = root if (root / "results").exists() else root.parent
        return data_root, code_root

    nested_code_root = root / "irt and k factors"
    if (nested_code_root / "src" / "disaster_irt").exists():
        return root, nested_code_root

    return root, root


def default_paths(project_root: Optional[Path] = None) -> dict:
    data_root, code_root = resolve_roots(project_root)
    return {
        "project_root": data_root,
        "code_root": code_root,
        "results_dir": data_root / "results",
        "irt_dir": data_root / "results" / "irt_baseline_ruleset",
        "output_dir": data_root / "analysis_outputs" / "irt_kfactor",
    }
