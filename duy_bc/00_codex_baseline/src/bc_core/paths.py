"""Stable project, repository, data, and output path resolution."""

from pathlib import Path


def baseline_root() -> Path:
    """Return the numbered baseline directory independent of process CWD."""
    return Path(__file__).resolve().parents[2]


def repository_root(config_path: Path | None = None) -> Path:
    """Return the repository containing `duy_bc` for a source or config path."""
    baseline = (
        baseline_root()
        if config_path is None
        else Path(config_path).expanduser().resolve().parent.parent
    )
    if baseline.parent.name != "duy_bc":
        raise ValueError(
            "relative corpus_root requires 00_codex_baseline inside a duy_bc directory"
        )
    return baseline.parent.parent.resolve()


def baseline_path(value: str | Path) -> Path:
    """Resolve generated/config paths relative to the numbered baseline."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (baseline_root() / path).resolve()


def corpus_path(value: str | Path, *, config_path: Path | None = None) -> Path:
    """Resolve replay data absolutely or relative to the containing repository."""
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (repository_root(config_path) / path).resolve()
    )
