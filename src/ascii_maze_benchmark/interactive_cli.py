import click
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from ascii_maze_benchmark.benchmark_runner import (
    benchmark_command,
    get_default_cache_dir,
)


@click.command(name="run-interactive")
def run_interactive_command() -> None:  # noqa: D401 – simple docstring ok for CLI
    """Run the ASCII‑maze benchmark in an *interactive* fashion.

    The command guides the user through a short questionnaire:

    1. Which model(s) to benchmark (comma‑separated list)
    2. Checks for ``OPENROUTER_API_KEY`` – if missing, prompts for it and can
       optionally persist the key into a local ``.env`` file so subsequent
       runs do not have to re‑enter it.
    3. Maze sizes, #mazes per size, RNG seed, and advanced flags such as
       *directional mode*, *verbose output*, *run models in parallel*, etc.

    After collecting all answers, the function calls the regular
    :pymeth:`ascii_maze_benchmark.benchmark_runner.benchmark_command` via
    ``ctx.invoke`` so the actual benchmark logic is executed in exactly the
    same way as if the user had typed the long CLI command manually.
    """

    ctx = click.get_current_context()

    # Load .env (and .venv if it is a *file* holding env variables rather than a
    # directory).  This allows users who keep their OPENROUTER_API_KEY inside a
    # local env‑file to have it picked up automatically before we start asking
    # questions.
    load_dotenv()  # does nothing if no file is present – safe to call always

    # ------------------------------------------------------------------
    # 1.  Model IDs
    # ------------------------------------------------------------------
    click.echo(
        click.style(
            "Browse available models at https://openrouter.ai/models", fg="cyan"
        )
    )

    default_models_msg = (
        "(comma‑separated, e.g. x-ai/grok-3-mini-beta,google/gemini-2.5-flash-preview)"
    )
    models_input = click.prompt(
        f"Enter model IDs {default_models_msg}",
        type=str,
    )
    model_ids: tuple[str, ...] = tuple(
        m.strip() for m in models_input.split(",") if m.strip()
    )

    # ------------------------------------------------------------------
    # 2.  API key handling
    # ------------------------------------------------------------------
    existing_key = os.getenv("OPENROUTER_API_KEY")

    if existing_key:
        if click.confirm(
            "OPENROUTER_API_KEY found in environment. Use this key?", default=True
        ):
            pass  # Keep existing key
        else:
            # User wants to override
            new_key = click.prompt(
                "Enter your OPENROUTER_API_KEY", hide_input=True, type=str
            )
            os.environ["OPENROUTER_API_KEY"] = new_key
            _maybe_persist_api_key(new_key)
    else:
        # Ask for the key outright (provide helpful link first)
        click.echo(
            click.style(
                "You can create a key at https://openrouter.ai/settings/keys",
                fg="cyan",
            )
        )
        new_key = click.prompt(
            "OPENROUTER_API_KEY not found – please enter it now",
            hide_input=True,
            type=str,
        )
        os.environ["OPENROUTER_API_KEY"] = new_key
        _maybe_persist_api_key(new_key)

    # ------------------------------------------------------------------
    # 3.  Benchmark parameters
    # ------------------------------------------------------------------
    maze_sizes_str = click.prompt(
        "Maze sizes to test (format: 3x3,4x4,5x5)",
        default="3x3,4x4,5x5,6x6",
        type=str,
    )

    mazes_per_size = click.prompt(
        "Number of different mazes to generate per size", default=3, type=int
    )

    seed = click.prompt("Random seed", default=42, type=int)

    directional_mode = click.confirm(
        "Directional mode? (model outputs a list of directions instead of the maze itself)",
        default=True,
    )

    verbose = click.confirm(
        "Verbose output (print maze + model response)?", default=False
    )

    parallel = False
    if len(model_ids) > 1:
        parallel = click.confirm(
            "More than one model specified – run benchmarks in parallel?", default=False
        )

    plot = click.confirm("Display results heat‑map after run?", default=True)
    plot_save: str | None = None
    if plot and click.confirm("Save heat‑map to file?", default=False):
        plot_save = click.prompt(
            "Enter file path to save image (leave blank to skip)", default="", type=str
        )
        if plot_save == "":
            plot_save = None

    # Cache directory – keep the default but allow override
    cache_dir_default = get_default_cache_dir()
    cache_dir = click.prompt(
        "Cache directory for API responses", default=cache_dir_default, type=str
    )

    # ------------------------------------------------------------------
    # 4.  Delegate to the actual benchmark command
    # ------------------------------------------------------------------
    ctx.invoke(
        benchmark_command,
        model_ids=model_ids,
        maze_sizes=maze_sizes_str,
        mazes_per_size=mazes_per_size,
        seed=seed,
        cache_dir=cache_dir,
        verbose=verbose,
        directional_mode=directional_mode,
        parallel=parallel,
        plot=plot,
        plot_save=plot_save,
    )


# ---------------------------------------------------------------------------
# Helper – optionally append / update the .env file with the provided key.
# ---------------------------------------------------------------------------


def _maybe_persist_api_key(api_key: str) -> None:
    """Offer to write/replace the OPENROUTER_API_KEY entry in a local .env file."""

    if not click.confirm("Save key to .env for future runs?", default=True):
        return

    env_path = Path(".env")

    # Read existing lines if file exists
    lines: list[str] = []
    if env_path.exists():
        try:
            lines = env_path.read_text().splitlines()
        except OSError:
            click.echo(
                click.style(
                    "Warning: could not read existing .env file – skipping save.",
                    fg="red",
                )
            )
            return

    # Remove any existing OPENROUTER_API_KEY line (case‑sensitive)
    pattern = re.compile(r"^OPENROUTER_API_KEY=.*$")
    lines = [ln for ln in lines if not pattern.match(ln)]

    # Append new key (no quoting so .env remains simple)
    lines.append(f"OPENROUTER_API_KEY={api_key}")

    try:
        env_path.write_text("\n".join(lines) + "\n")
        click.echo(click.style("Saved key to .env", fg="green"))
    except OSError as exc:
        click.echo(
            click.style(
                f"Warning: failed to write .env file ({exc}). Key not saved.", fg="red"
            )
        )
