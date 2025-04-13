import click

from ascii_maze_benchmark.generate_maze_script import generate_maze_command
from ascii_maze_benchmark.benchmark_runner import benchmark_command


@click.group()
def cli():
    """ASCII Maze Benchmark - Tools for testing LLM performance on ASCII mazes."""
    pass


cli.add_command(generate_maze_command)
cli.add_command(benchmark_command)
