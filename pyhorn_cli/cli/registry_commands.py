"""Registry CLI commands for pyhorn."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from pyhorn_registry import Registry, registry

app = typer.Typer(help="Manage the pyhorn driver and project registry")


@app.command(name="list")
def list_items(
    kind: Optional[str] = typer.Option(
        None, "--kind", "-k", help="Filter by 'driver' or 'project'"
    ),
    base: Optional[Path] = typer.Option(
        None, "--base", help="Registry base directory (default: ~/.pyhorn)"
    ),
):
    """List all registered drivers and projects."""
    reg = registry(base=base)
    entries = reg.list(kind=kind)
    if not entries:
        kind_label = f" {kind}" if kind else ""
        typer.secho(f"No entries{kind_label} in registry.", fg="yellow")
        return

    for e in entries:
        tags = f" [{', '.join(e.tags)}]" if e.tags else ""
        desc = f" — {e.description}" if e.description else ""
        typer.secho(f"  {e.name} ({e.kind}){tags}{desc}")


@app.command()
def get(name: str, base: Optional[Path] = None):
    """Show details for a single entry."""
    reg = registry(base=base)
    e = reg.get(name)
    if e is None:
        raise typer.Exit(code=1, err=True)
    path = reg.resolve_path(name)
    typer.secho(f"Name:       {e.name}", bold=True)
    typer.secho(f"Kind:       {e.kind}")
    typer.secho(f"File:       {path}")
    if e.description:
        typer.secho(f"Description:{e.description}")
    if e.tags:
        typer.secho(f"Tags:       {', '.join(e.tags)}")
    typer.secho(f"Created:    {e.created}")
    typer.secho(f"Modified:   {e.modified}")


@app.command()
def add(
    name: str,
    kind: str = typer.Argument(..., help="'driver' or 'project'"),
    file: Path = typer.Option(..., "--file", "-f", help="Path to source YAML file"),
    description: str = typer.Option("", "--description", help="Entry description"),
    tag: list[str] = typer.Option([], "--tag", help="Tags to attach (repeatable)"),
    base: Optional[Path] = typer.Option(None, "--base", help="Registry base dir"),
    reference: bool = typer.Option(
        False, "--reference", help="Store file path reference instead of copying"
    ),
):
    """Add a driver or project to the registry."""
    if kind not in ("driver", "project"):
        typer.secho("kind must be 'driver' or 'project'", fg="red")
        raise typer.Exit(code=1)
    if not file.exists():
        typer.secho(f"File not found: {file}", fg="red")
        raise typer.Exit(code=1)

    reg = registry(base=base)
    try:
        entry = reg.add(
            name=name,
            kind=kind,
            source_path=file,
            description=description,
            tags=tag,
            copy=not reference,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(code=1)

    typer.secho(f"Added '{name}' ({kind})", fg="green")


@app.command()
def remove(
    name: str,
    base: Optional[Path] = None,
    delete_file: bool = typer.Option(
        False, "--delete-file", help="Also delete the underlying YAML file"
    ),
):
    """Remove an entry from the registry."""
    reg = registry(base=base)
    try:
        reg.remove(name, delete_file=delete_file)
    except KeyError:
        typer.secho(f"No entry named '{name}'", fg="red")
        raise typer.Exit(code=1)
    typer.secho(f"Removed '{name}'" + (" (file deleted)" if delete_file else ""), fg="green")


@app.command()
def update(
    name: str,
    base: Optional[Path] = None,
    description: Optional[str] = typer.Option(None, "--description"),
    add_tag: list[str] = typer.Option([], "--add-tag", help="Add tags (repeatable)"),
    remove_tag: list[str] = typer.Option([], "--remove-tag", help="Remove tags (repeatable)"),
):
    """Update metadata (description, tags) for an entry."""
    reg = registry(base=base)
    e = reg.get(name)
    if e is None:
        typer.secho(f"No entry named '{name}'", fg="red")
        raise typer.Exit(code=1)

    new_tags = None
    if add_tag or remove_tag:
        current = set(e.tags)
        current.update(add_tag)
        current.difference_update(remove_tag)
        new_tags = sorted(current)

    try:
        updated = reg.update_metadata(
            name,
            description=description,
            tags=new_tags,
        )
    except KeyError:
        typer.secho(f"No entry named '{name}'", fg="red")
        raise typer.Exit(code=1)

    typer.secho(f"Updated '{name}'", fg="green")


@app.command()
def resolve(name: str, base: Optional[Path] = None):
    """Show the file path for a registered entry."""
    reg = registry(base=base)
    path = reg.resolve_path(name)
    if path is None:
        typer.secho(f"No entry named '{name}'", fg="red")
        raise typer.Exit(code=1)
    typer.secho(str(path))


@app.command()
def load(name: str, base: Optional[Path] = None):
    """Load and print the YAML file for a registered entry."""
    reg = registry(base=base)
    try:
        data = reg.load_yaml(name)
    except FileNotFoundError:
        typer.secho(f"No file found for '{name}'", fg="red")
        raise typer.Exit(code=1)
    print(yaml.dump(data, default_flow_style=False, sort_keys=False))


@app.command()
def import_existing(
    kind: str = typer.Argument(..., help="'driver' or 'project'"),
    base: Optional[Path] = None,
):
    """Import all existing YAML files from drivers/ and projects/ directories into the registry.

    Scans the local drivers/ and projects/ directories in the current working directory
    and registers every .yaml file found, using the filename (without extension) as the entry name.
    Skips entries that are already registered.
    """
    if kind not in ("driver", "project"):
        typer.secho("kind must be 'driver' or 'project'", fg="red")
        raise typer.Exit(code=1)

    reg = registry(base=base)
    subdir = {"driver": "drivers", "project": "projects"}[kind]
    src_dir = Path.cwd() / subdir

    if not src_dir.is_dir():
        typer.secho(f"Directory not found: {src_dir}", fg="yellow")
        return

    imported = 0
    skipped = 0
    for yaml_file in sorted(src_dir.glob("*.yaml")):
        name = yaml_file.stem
        if reg.exists(name):
            skipped += 1
            continue
        reg.add(name=name, kind=kind, source_path=yaml_file, copy=False)
        typer.secho(f"  + {name}", fg="green")
        imported += 1

    typer.secho(
        f"Imported {imported} entries, skipped {skipped} (already registered).",
        fg="green" if imported else None,
    )


if __name__ == "__main__":
    app()