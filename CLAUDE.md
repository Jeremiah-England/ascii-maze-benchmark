# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install dependencies: `uv add [package-name]`
- Build: `uv pip install -e .`
- Run CLI app: `uv run ascii-maze-benchmark [COMMAND] [ARGS]`
- Run generate-example: `uv run ascii-maze-benchmark generate-example [WIDTH] [HEIGHT] [--seed SEED]`
- Run module directly: `uv run python -m ascii_maze_benchmark`
- Lint/Format/Type check/Test: `uv run pre-commit run --all-files`
  - These pre-commit hooks include running pyright and pytest.

## Project Structure

- CLI commands are implemented using Click
- The main CLI entry point is defined in `__init__.py`
- Individual commands are organized as subcommands

## Coding Conventions

- Python 3.13+ compatibility required
- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines (enforced by ruff)
- Use Google-style docstrings with proper Args/Returns sections
- Use pathlib for file operations
- Store benchmark results in cache to avoid unnecessary API calls
- Use pytest fixtures for test setup
- OpenRouter API credentials should be stored in a .env file
