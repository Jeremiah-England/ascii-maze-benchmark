"""ASCII Maze Benchmark package – CLI definitions."""

from __future__ import annotations

import click

from ascii_maze_benchmark.benchmark_runner import benchmark_command
from ascii_maze_benchmark.generate_maze_script import (
    generate_maze_command,
    solve_maze_command,
)


@click.group()
def cli() -> None:  # noqa: D401 – simple docstring ok for CLI
    """Command‑line interface for the ASCII Maze benchmark."""


# Register sub‑commands
cli.add_command(generate_maze_command)
cli.add_command(solve_maze_command)
cli.add_command(benchmark_command)
