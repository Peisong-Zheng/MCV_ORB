"""Local output location for generated experiment notes."""

from pathlib import Path


def generated_notes_dir(output_root):
    """Keep notes outside research outputs, including in isolated review runs.

    Callers supply their catalogue output directory. Source-catalogue notes
    share the parent workspace's agent_work folder with primary-catalogue notes.
    """
    root = Path(output_root)
    scope = "primary"
    if root.name in {"NGRIP", "MIS6", "Barker2011"}:
        scope, root = root.name, root.parent
    return root / "agent_work/scratch/experiment_note" / scope
