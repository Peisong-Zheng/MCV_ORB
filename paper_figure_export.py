"""Copy the current manuscript's selected PDFs into monsoon_paper/figures.

The explicit mapping is in monsoon_paper/figure_manifest.csv. Analysis
exporters call copy_pdf_to_paper after saving their canonical PDF; running
this script refreshes all available mapped figures without rerunning fits.
Planned figures remain pending until their new source PDFs are produced.
"""

from pathlib import Path
import argparse
import csv
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = Path("monsoon_paper/figure_manifest.csv")


def figure_manifest(project_root=PROJECT_ROOT):
    """Read the paper's single source-to-destination mapping."""
    with (Path(project_root) / MANIFEST_PATH).open(newline="") as handle:
        return list(csv.DictReader(handle))


def paper_path(entry, project_root=PROJECT_ROOT):
    return Path(project_root) / "monsoon_paper/figures" / entry["paper_name"]


def copy_pdf_to_paper(source_pdf, project_root=PROJECT_ROOT):
    """Copy only a mapped canonical source; ignore temporary or other exports."""
    project_root = Path(project_root).resolve()
    source_pdf = Path(source_pdf).resolve()
    for entry in figure_manifest(project_root):
        if source_pdf == (project_root / entry["source_relpath"]).resolve():
            target = paper_path(entry, project_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_pdf, target)
            return target
    return None


def export_all(project_root=PROJECT_ROOT):
    """Refresh available figures, leaving planned slots and unrelated files alone."""
    project_root = Path(project_root)
    entries = figure_manifest(project_root)
    missing = [entry["source_relpath"] for entry in entries
               if entry["status"] != "planned"
               and not (project_root / entry["source_relpath"]).is_file()]
    if missing:
        raise FileNotFoundError("Missing ready figure source(s): " + ", ".join(missing))
    written = []
    for entry in entries:
        source = project_root / entry["source_relpath"]
        if source.is_file():
            target = paper_path(entry, project_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            written.append(target)
    return written


def sync_status(project_root=PROJECT_ROOT):
    """Report missing, pending, stale, and byte-identical paper copies."""
    project_root = Path(project_root)
    rows = []
    for entry in figure_manifest(project_root):
        source = project_root / entry["source_relpath"]
        target = paper_path(entry, project_root)
        if not source.is_file():
            state = "planned" if entry["status"] == "planned" else "missing source"
            if target.exists():
                state = "orphan copy (source missing)"
        elif not target.is_file():
            state = "not copied"
        elif source.read_bytes() != target.read_bytes():
            state = "out of date"
        else:
            state = "synced"
        rows.append({**entry, "sync_status": state})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="List figure status without copying.")
    mode.add_argument("--check", action="store_true", help="Check available copies; exit 1 if stale or missing.")
    args = parser.parse_args()
    if not (args.status or args.check):
        export_all()
    rows = sync_status()
    for row in rows:
        print(f"{row['paper_name']:12} {row['sync_status']:28} {row['source_relpath']}")
    pending = sum(row["sync_status"] == "planned" for row in rows)
    print(f"{sum(row['sync_status'] == 'synced' for row in rows)} synced; {pending} planned.")
    if args.check and any(row["sync_status"] not in {"synced", "planned"} for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
