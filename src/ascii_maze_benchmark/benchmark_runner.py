import os
import random
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Union
import time

import click
import requests
from dotenv import load_dotenv
from diskcache import Cache

from ascii_maze_benchmark.generate_maze_script import (
    generate_maze,
    solve_maze,
    solution_to_directions,
)

# Regex to strip ANSI escape sequences for styling
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class BenchmarkRunner:
    """Class to handle running the ASCII maze benchmark against various models."""

    def __init__(
        self,
        model_id: str,
        cache_dir: str = ".cache/api_responses",
        verbose: bool = False,
        directional_mode: bool = False,
    ):
        """
        Initialize the benchmark runner.

        Args:
            model_id: The OpenRouter model ID to test
            cache_dir: Directory to cache API responses
            verbose: Whether to print detailed information during benchmarking
            directional_mode: If True, tests the model's ability to output directional
                              instructions instead of marking the maze with dots
        """
        self.verbose = verbose
        self.directional_mode = directional_mode
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Please add it to your .env file."
            )

        self.model_id = model_id

        # Create cache directory if it doesn't exist
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        # Cache for LLM API responses
        self.api_cache = Cache(directory=str(cache_dir_path))

    def run_benchmark(
        self, maze_sizes: List[tuple], num_mazes_per_size: int = 3, seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run the benchmark on the specified model.

        Args:
            maze_sizes: List of tuples (width, height) for maze sizes to test
            num_mazes_per_size: Number of different mazes to generate per size
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing benchmark results
        """
        # Create a local random number generator for reproducibility and thread safety
        rng = random.Random(seed)

        results = {
            "model_id": self.model_id,
            "maze_sizes": maze_sizes,
            "num_mazes_per_size": num_mazes_per_size,
            "seed": seed,
            "timestamp": time.time(),
            "results": [],
        }

        maze_seeds = [rng.randint(1, 10000) for _ in range(num_mazes_per_size)]

        total_mazes = len(maze_sizes) * num_mazes_per_size
        current_maze = 0

        for width, height in maze_sizes:
            for maze_seed in maze_seeds:
                current_maze += 1
                # Use bright blue for the maze number and cyan for dimensions
                click.echo(
                    f"Testing maze {click.style(f'{current_maze}/{total_mazes}', fg='bright_blue')}: {click.style(f'{width}x{height}', fg='cyan')} (seed: {click.style(str(maze_seed), fg='cyan')})"
                )

                # Generate maze
                maze = generate_maze(width, height, maze_seed)
                if not maze:
                    click.echo(
                        click.style(
                            f"Failed to generate maze of size {width}x{height}",
                            fg="red",
                        )
                    )
                    continue

                # Initialize directions variable
                directions = []

                # Solve maze to get the correct solution
                if self.directional_mode:
                    solution, raw_path = solve_maze(maze, return_raw_path=True)
                    if not solution or not raw_path:
                        click.echo(
                            click.style(
                                f"Failed to solve maze of size {width}x{height}",
                                fg="red",
                            )
                        )
                        continue
                    # Convert raw path to directions
                    directions = solution_to_directions(raw_path)
                else:
                    solution = solve_maze(maze)
                    if not solution:
                        click.echo(
                            click.style(
                                f"Failed to solve maze of size {width}x{height}",
                                fg="red",
                            )
                        )
                        continue

                # Convert maze to string for prompt
                maze_str = "\n".join(
                    ["START", " v"]
                    + maze
                    + [
                        " " * (len(maze[0]) - 2) + "^",
                        " " * (len(maze[0]) - 6) + "FINISH",
                    ]
                )

                # Create prompt based on mode
                if self.directional_mode:
                    prompt = f"""
Below is an ASCII maze. The start is at the top and marked with 'START' and the finish is at the bottom and marked with 'FINISH'.
Your task is to find a valid path through the maze from start to finish.

In the solution, provide a comma-separated list of directions: "up", "down", "left", "right".

You can think about your approach and summarize your reasoning as much as you wish.
However, your final solution must be enclosed within a ```solution{{}}``` code block.

For example, if given this:

START
 v
# #########
#   #     #
# # # ### #
# #     # #
# ##### # #
# #   # # #
# ### ### #
#   #   # #
### ### # #
#     #   #
######### #
         ^
     FINISH

Your output should be:

```solution
down,right,right,down,down,right,right,up,up,right,right,right,right,down,down,down,down,down,down,down,down,down
```

Here's the maze:

{maze_str}
"""
                else:
                    prompt = f"""
Below is an ASCII maze. The start is at the top and marked with 'START' and the finish is at the bottom and marked with 'FINISH'.
Your task is to find a valid path through the maze from start to finish.

In the solution, mark the path using periods ('.') for each step of the path.
Do not include the START and FINISH labels or arrows in your solution.

You can think about your approach and summarize your reasoning as much as you wish.
However, your final solution must be enclosed within a ```solution{{}}``` code block.

For example, if given this:

START
 v
# #########
#   #     #
# # # ### #
# #     # #
# ##### # #
# #   # # #
# ### ### #
#   #   # #
### ### # #
#     #   #
######### #
         ^
     FINISH

Your output should be:

```solution
#.#########
#...#.....#
# #.#.###.#
# #...  #.#
# ##### #.#
# #   # #.#
# ### ###.#
#   #   #.#
### ### #.#
#     #  .#
#########.#
```

Here's the maze:

{maze_str}
"""

                if self.verbose:
                    click.echo("\n=== Input Maze ===")
                    click.echo(click.style(maze_str, fg="cyan", bold=True))
                    click.echo("==================\n")

                # Call the API to get the model's solution
                response = self._call_openrouter_api(prompt)
                model_solution = self._extract_solution(response)

                # Evaluate the solution based on mode
                if self.directional_mode:
                    # Compare directions
                    exact_match = self._is_exact_match(directions, model_solution)

                    # Don't use Levenshtein distance for directional mode
                    # Set dummy values for the result format compatibility
                    levenshteins = {
                        "exact": exact_match,
                        "distance_1": exact_match,
                        "distance_2": exact_match,
                        "distance_5": exact_match,
                        "distance_10": exact_match,
                        "distance_20": exact_match,
                        "percent_95": exact_match,
                        "percent_90": exact_match,
                        "percent_80": exact_match,
                    }
                    distance = 0 if exact_match else 999

                    # Print comparison if verbose
                    if self.verbose and model_solution:
                        click.echo("\n=== Correct Directions vs Model Directions ===")

                        match_description = "Exact match"
                        if exact_match and directions != model_solution and directions:
                            if (
                                directions[0] == "down"
                                and model_solution == directions[1:]
                            ):
                                match_description = "Match (first 'down' omitted)"
                            elif (
                                directions[-1] == "down"
                                and model_solution == directions[:-1]
                            ):
                                match_description = "Match (last 'down' omitted)"
                            elif (
                                len(model_solution) >= 1
                                and model_solution[-1] == "down"
                                and directions == model_solution[:-1]
                            ):
                                match_description = (
                                    "Match (model provided extra 'down')"
                                )
                            elif (
                                len(directions) >= 2
                                and directions[0] == "down"
                                and directions[-1] == "down"
                                and model_solution == directions[1:-1]
                            ):
                                match_description = (
                                    "Match (first and last 'down' omitted)"
                                )
                            elif (
                                len(model_solution) >= 2
                                and model_solution[-1] == "down"
                                and model_solution[-2] == "down"
                                and directions == model_solution[:-2]
                            ):
                                match_description = (
                                    "Match (model provided two extra 'downs')"
                                )
                            elif (
                                len(model_solution) >= 2
                                and model_solution[0] == "down"
                                and model_solution[-1] == "down"
                                and model_solution[1:-1] == directions
                            ):
                                match_description = (
                                    "Match (model provided two extra 'downs')"
                                )

                        click.echo(
                            f"{match_description}: {click.style('✓', fg='green', bold=True) if exact_match else click.style('✗', fg='red', bold=True)}"
                        )
                        click.echo("\nCorrect directions:")
                        click.echo(click.style(",".join(directions), fg="green"))
                        click.echo("\nModel directions:")
                        click.echo(click.style(",".join(model_solution), fg="yellow"))
                        click.echo("===========================================\n")
                else:
                    # Compare maze solutions
                    exact_match = self._is_exact_match(solution, model_solution)
                    levenshteins = self._calculate_levenshteins(
                        solution, model_solution
                    )

                    # Print comparison if verbose
                    if self.verbose and model_solution:
                        click.echo("\n=== Correct Solution vs Model Solution ===")
                        # Calculate Levenshtein distance for display
                        distance = self._levenshtein_distance(
                            "\n".join(solution), "\n".join(model_solution)
                        )
                        click.echo(
                            f"Exact match: {click.style('✓', fg='green', bold=True) if exact_match else click.style('✗', fg='red', bold=True)} (Levenshtein distance: {click.style(str(distance), fg='cyan', bold=True)})"
                        )
                        click.echo("\nCorrect solution:")
                        click.echo(click.style("\n".join(solution), fg="green"))
                        click.echo("\nModel solution:")
                        click.echo(click.style("\n".join(model_solution), fg="yellow"))
                        click.echo("=========================================\n")

                    # Calculate Levenshtein distance for result
                    distance = self._levenshtein_distance(
                        "\n".join(solution), "\n".join(model_solution)
                    )

                # Display result
                match_text = "✓" if exact_match else "✗"
                if self.directional_mode:
                    click.echo(
                        f"Solution: {click.style(match_text, fg='green' if exact_match else 'red', bold=True)}"
                    )
                else:
                    click.echo(
                        f"Solution: {click.style(match_text, fg='green' if exact_match else 'red', bold=True)} (Levenshtein distance: {click.style(str(distance), fg='cyan', bold=True)})"
                    )

                # Store result
                if self.directional_mode:
                    result = {
                        "maze_size": (width, height),
                        "maze_seed": maze_seed,
                        "correct_solution": solution,
                        "correct_directions": directions,
                        "model_solution": model_solution,
                        "exact_match": exact_match,
                        "levenshteins": levenshteins,
                        "levenshtein_distance": distance,
                        "mode": "directional",
                    }
                else:
                    result = {
                        "maze_size": (width, height),
                        "maze_seed": maze_seed,
                        "correct_solution": solution,
                        "model_solution": model_solution,
                        "exact_match": exact_match,
                        "levenshteins": levenshteins,
                        "levenshtein_distance": distance,
                        "mode": "maze",
                    }

                results["results"].append(result)

        return results

    def _call_openrouter_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the OpenRouter API with the provided prompt.

        Uses a cache to avoid redundant API calls for the same prompt and model.
        """
        # Create a cache key based on the model and prompt
        # We use SHA-256 to handle very long prompts that might exceed key length limits
        key = f"{self.model_id}:{hashlib.sha256(prompt.encode()).hexdigest()}"

        # Check if we have a cached response
        cached_response = self.api_cache.get(key)
        if cached_response is not None:
            if self.verbose:
                click.echo(
                    click.style(
                        "Using cached LLM API response", fg="bright_black", italic=True
                    )
                )
            assert isinstance(cached_response, dict)
            return cached_response

        # If not in cache, make the API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.verbose:
            click.echo(
                click.style(
                    "Calling OpenRouter API (not cached)", fg="bright_blue", italic=True
                )
            )

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data
        )

        if response.status_code != 200:
            raise Exception(
                f"API call failed with status code {response.status_code}: {response.text}"
            )

        # Cache the response
        result = response.json()
        self.api_cache.set(key, result)

        return result

    @staticmethod
    def extract_solution_from_content(content: str) -> List[str]:
        """
        Extract maze solution from model response content.

        Args:
            content: The text content from the model response

        Returns:
            List of strings representing the maze solution
        """
        # Look for ```solution code blocks - find the LAST one if multiple exist
        solution_blocks = []
        current_block = []
        in_solution_block = False
        lines = content.strip().split("\n")

        for line in lines:
            line_stripped = line.strip()
            # Check for ```solution block start
            if line_stripped.startswith("```solution"):
                in_solution_block = True
                current_block = []
                continue
            # Check for block end if we're in a solution block
            elif in_solution_block and line_stripped.startswith("```"):
                in_solution_block = False
                # Only keep blocks that have maze-like content
                if any("#" in line or "." in line for line in current_block):
                    solution_blocks.append(current_block)
                continue
            # Collect lines within solution block
            elif in_solution_block and line_stripped and ("#" in line or "." in line):
                current_block.append(line_stripped)

        # If we found any solution blocks, return the last one
        if solution_blocks:
            return solution_blocks[-1]

        # If no solution blocks found, we ignore all regular code blocks
        # Skip directly to raw text parsing

        # We should only return solution blocks, so return empty list if none found
        return []

    @staticmethod
    def extract_directions_from_content(content: str) -> List[str]:
        """
        Extract directional instructions from model response content.

        Args:
            content: The text content from the model response

        Returns:
            List of directional instructions (up, down, left, right)
        """
        # Look for ```solution code blocks - find the LAST one if multiple exist
        solution_blocks = []
        current_block = []
        in_solution_block = False
        lines = content.strip().split("\n")

        for line in lines:
            line_stripped = line.strip()
            # Check for ```solution block start
            if line_stripped.startswith("```solution"):
                in_solution_block = True
                current_block = []
                continue
            # Check for block end if we're in a solution block
            elif in_solution_block and line_stripped.startswith("```"):
                in_solution_block = False
                # Only keep blocks that have direction-like content
                if current_block:
                    solution_blocks.append("\n".join(current_block))
                continue
            # Collect lines within solution block
            elif in_solution_block:
                current_block.append(line_stripped)

        # If we found any solution blocks, parse the last one
        if solution_blocks:
            directions_text = solution_blocks[-1]

            # Extract all directional words using regex
            direction_pattern = r"\b(up|down|left|right)\b"
            directions = re.findall(direction_pattern, directions_text.lower())

            if directions:
                return directions

        # Fall back to trying to find directions in the whole text
        # Look for patterns like "up, down, left, right" or "up,down,left,right"
        direction_pattern = r"\b(up|down|left|right)\b"
        directions = re.findall(direction_pattern, content.lower())

        return directions

    def _extract_solution(self, response: Dict[str, Any]) -> List[str]:
        """Extract the solution from the API response."""
        try:
            # Extract the content from the assistant's message
            content = response["choices"][0]["message"]["content"]

            if self.verbose:
                click.echo("\n=== Model Response ===")
                click.echo(click.style(content, fg="bright_black"))
                click.echo("=====================\n")

            if self.directional_mode:
                return self.extract_directions_from_content(content)
            else:
                return self.extract_solution_from_content(content)

        except (KeyError, IndexError):
            return []

    def _is_exact_match(
        self, correct_solution: List[str], model_solution: List[str]
    ) -> bool:
        """
        Check if the model's solution exactly matches the correct solution.

        In directional mode, we also check if the solution is correct after
        dropping a "down" from either or both ends when evaluating directional solutions.
        """
        # Handle empty solutions
        if not correct_solution or not model_solution:
            # In directional mode, if correct solution is just a single "down",
            # and model solution is empty, consider it a match
            if (
                self.directional_mode
                and correct_solution == ["down"]
                and not model_solution
            ):
                return True
            return correct_solution == model_solution

        # Standard exact match check
        if correct_solution == model_solution:
            return True

        # For directional mode, perform additional checks
        if self.directional_mode and "down" in correct_solution:
            # Check with "down" removed from the beginning if it exists
            if correct_solution[0] == "down" and model_solution == correct_solution[1:]:
                return True

            # Check with "down" removed from the end if it exists
            if (
                correct_solution[-1] == "down"
                and model_solution == correct_solution[:-1]
            ):
                return True

            # Check if model provided an extra "down" at the end
            if (
                len(model_solution) >= 1
                and model_solution[-1] == "down"
                and correct_solution == model_solution[:-1]
            ):
                return True

            # Check if model provided two extra "downs" at the end
            if (
                len(model_solution) >= 2
                and model_solution[-1] == "down"
                and model_solution[-2] == "down"
                and correct_solution == model_solution[:-2]
            ):
                return True

            # Check with "down" removed from both ends if they exist
            if (
                len(correct_solution) >= 2
                and correct_solution[0] == "down"
                and correct_solution[-1] == "down"
                and model_solution == correct_solution[1:-1]
            ):
                return True

            # Check if model provided an extra "down" after removing first "down"
            if (
                len(model_solution) >= 2
                and correct_solution[0] == "down"
                and model_solution[-1] == "down"
                and correct_solution[1:] == model_solution[:-1]
            ):
                return True

            # Check if model provided two extra "downs" after removing first "down"
            if (
                len(model_solution) >= 3
                and correct_solution[0] == "down"
                and model_solution[-1] == "down"
                and model_solution[-2] == "down"
                and correct_solution[1:] == model_solution[:-2]
            ):
                return True

            # Check if model provided an extra "down" at both the front and end
            if (
                len(model_solution) >= 2
                and model_solution[0] == "down"
                and model_solution[-1] == "down"
                and model_solution[1:-1] == correct_solution
            ):
                return True

        return False

    def _calculate_levenshteins(
        self,
        correct_solution: Union[List[str], str],
        model_solution: Union[List[str], str],
    ) -> Dict[str, bool]:
        """
        Calculate Levenshtein distances for different thresholds.

        Args:
            correct_solution: Either a list of strings or a single string with the correct solution
            model_solution: Either a list of strings or a single string with the model's solution

        Returns:
            Dictionary of thresholds and whether they are met
        """
        # Convert to strings if they're lists
        if isinstance(correct_solution, list):
            correct_str = "\n".join(correct_solution)
        else:
            correct_str = correct_solution

        if isinstance(model_solution, list):
            model_str = "\n".join(model_solution)
        else:
            model_str = model_solution

        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(correct_str, model_str)

        # Check if within various thresholds
        thresholds = {
            "exact": distance == 0,
            "distance_1": distance <= 1,
            "distance_2": distance <= 2,
            "distance_5": distance <= 5,
            "distance_10": distance <= 10,
            "distance_20": distance <= 20,
            "percent_95": distance <= len(correct_str) * 0.05,
            "percent_90": distance <= len(correct_str) * 0.1,
            "percent_80": distance <= len(correct_str) * 0.2,
        }

        return thresholds

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


@click.command(name="run-benchmark")
@click.argument("model_ids", nargs=-1, type=str)
@click.option(
    "--maze-sizes",
    type=str,
    default="3x3,4x4,5x5,6x6",
    help="Comma-separated list of maze sizes to test (format: WIDTHxHEIGHT)",
)
@click.option(
    "--mazes-per-size",
    type=int,
    default=3,
    help="Number of different mazes to generate per size",
)
@click.option(
    "--seed", type=int, default=42, help="Random seed for reproducible maze generation"
)
@click.option(
    "--cache-dir",
    type=str,
    default=".cache/api_responses",
    help="Directory to cache API responses",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print the input maze and model's response for each test",
)
@click.option(
    "--directional-mode",
    is_flag=True,
    help="Test model's ability to output directional instructions instead of marking the maze",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Run benchmarks for multiple models in parallel (one thread per model)",
)
@click.option(
    "--plot",
    is_flag=True,
    help="Display a matplotlib heat‑map of the results after the run (interactive)",
)
@click.option(
    "--plot-save",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Additionally save the heat‑map to the given file path",
)
def benchmark_command(
    model_ids: tuple[str, ...],
    maze_sizes: str,
    mazes_per_size: int,
    seed: int,
    cache_dir: str,
    verbose: bool = False,
    directional_mode: bool = False,
    parallel: bool = False,
    plot: bool = False,
    plot_save: str | None = None,
):
    """Run ASCII maze benchmark on the specified model."""
    # Parse maze sizes from string (format: "3x3,4x4,5x5")
    try:
        sizes = []
        for size in maze_sizes.split(","):
            width, height = map(int, size.split("x"))
            sizes.append((width, height))
    except ValueError:
        click.echo(
            click.style(
                "Error: Invalid maze sizes format. Use format: WIDTHxHEIGHT,WIDTHxHEIGHT",
                fg="red",
                bold=True,
            )
        )
        return

    # Ensure at least one model ID is provided
    if not model_ids:
        click.echo(
            click.style(
                "Error: At least one model ID must be provided.", fg="red", bold=True
            )
        )
        return

    # ------------------------------------------------------------------
    # Unified benchmarking logic
    # Instead of maintaining two largely identical branches for the single
    # ‑model and multi‑model cases, we now treat both scenarios uniformly by
    # always iterating over the supplied ``model_ids`` tuple.  At the end we
    # optionally print a comparison table if more than one model was
    # benchmarked.
    # ------------------------------------------------------------------

    def _print_run_header(curr_model_id: str) -> None:
        """Prints a banner with basic run parameters."""

        click.echo(
            "\n"
            + click.style(f"Running benchmark on model: {curr_model_id}", bold=True)
        )
        click.echo(f"Testing maze sizes: {click.style(maze_sizes, fg='cyan')}")
        click.echo(f"Mazes per size: {click.style(str(mazes_per_size), fg='cyan')}")

        mode_label = "Directional" if directional_mode else "Maze Marking"
        mode_color = "magenta" if directional_mode else "blue"
        click.echo(f"Mode: {click.style(mode_label, fg=mode_color, bold=True)}")

        if verbose:
            click.echo(f"Verbose mode: {click.style('ON', fg='green', bold=True)}")

    def _print_model_summary(curr_model_id: str, run_results: Dict[str, Any]) -> dict:
        """Prints a detailed summary for one model and returns per‑size stats."""

        exact_matches = sum(r["exact_match"] for r in run_results["results"])
        total = len(run_results["results"])
        percentage = exact_matches / total * 100 if total else 0.0

        if percentage == 100:
            overall_color = "green"
        elif percentage >= 66.6667:
            overall_color = "yellow"
        elif percentage >= 33.3333:
            overall_color = "bright_yellow"
        else:
            overall_color = "red"

        click.echo("\n" + click.style("Benchmark Results Summary:", bold=True))
        click.echo(f"Model: {click.style(curr_model_id, fg='bright_blue', bold=True)}")

        if directional_mode:
            click.echo(
                f"Mode: {click.style('Directional Instructions', fg='magenta', bold=True)}"
            )
            click.echo(
                f"Matching: {click.style('Allowing omission of leading "down" and accepting up to two extra "down" commands at the end', fg='cyan')}"
            )
            click.echo(
                f"Exact matches: {click.style(f'{exact_matches}/{total} ({percentage:.2f}%)', fg=overall_color, bold=True)}"
            )
        else:
            avg_distance = (
                sum(r["levenshtein_distance"] for r in run_results["results"]) / total
                if total
                else 0.0
            )
            click.echo(f"Mode: {click.style('Maze Marking', fg='blue', bold=True)}")
            click.echo(
                f"Exact matches: {click.style(f'{exact_matches}/{total} ({percentage:.2f}%)', fg=overall_color, bold=True)} (Avg Levenshtein distance: {click.style(f'{avg_distance:.2f}', fg='cyan', bold=True)})"
            )

            thresholds = [
                "exact",
                "distance_1",
                "distance_2",
                "distance_5",
                "distance_10",
                "distance_20",
                "percent_95",
                "percent_90",
                "percent_80",
            ]
            for threshold in thresholds:
                matches_t = sum(
                    1 for r in run_results["results"] if r["levenshteins"][threshold]
                )
                match_pct = matches_t / total * 100 if total else 0.0
                thresh_color = (
                    "green"
                    if match_pct >= 75
                    else "yellow"
                    if match_pct >= 50
                    else "red"
                )
                disp = threshold.replace("_", " ").replace("percent", "within").title()
                click.echo(
                    f"{disp}: {click.style(f'{matches_t}/{total} ({match_pct:.2f}%)', fg=thresh_color)}"
                )

        # Per‑size summary
        click.echo()
        click.echo(click.style("Per-Size Summary:", bold=True))

        per_size_stats: dict[tuple[int, int], tuple[int, int]] = {}
        for w, h in sorted(sizes):
            size_res = [r for r in run_results["results"] if r["maze_size"] == (w, h)]
            cnt = len(size_res)
            mat = sum(r["exact_match"] for r in size_res)
            pct = mat / cnt * 100 if cnt else 0.0
            per_size_stats[(w, h)] = (mat, cnt)

            if pct == 100:
                col = "green"
            elif pct >= 66.6667:
                col = "yellow"
            elif pct >= 33.3333:
                col = "bright_yellow"
            else:
                col = "red"

            if directional_mode:
                click.echo(
                    f"{w}x{h}: {click.style(f'{mat}/{cnt} ({pct:.2f}%)', fg=col)}"
                )
            else:
                avg_dist = (
                    sum(r["levenshtein_distance"] for r in size_res) / cnt
                    if cnt
                    else 0.0
                )
                click.echo(
                    f"{w}x{h}: {click.style(f'{mat}/{cnt} ({pct:.2f}%)', fg=col)} (Avg Levenshtein distance: {click.style(f'{avg_dist:.2f}', fg='cyan')})"
                )

        return per_size_stats

    # ------- Execute benchmarks for each requested model -------

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Helper that runs benchmark for a single model and returns per‑size stats.
    def _run_model(curr_model_id: str) -> dict[tuple[int, int], tuple[int, int]]:
        """Wrapper to execute benchmark for one model; meant for ThreadPoolExecutor."""

        # Ensure each model prints its header at the start of the run so the user
        # gets immediate feedback even when running in parallel.
        _print_run_header(curr_model_id)

        try:
            runner = BenchmarkRunner(
                curr_model_id, cache_dir, verbose, directional_mode
            )
            run_results = runner.run_benchmark(sizes, mazes_per_size, seed)
        except Exception as exc:
            click.echo(
                click.style(
                    f"Error running benchmark for model {curr_model_id}: {exc}",
                    fg="red",
                    bold=True,
                )
            )
            # Return empty stats so downstream logic does not fail.
            return {}

        # Print model summary (thread‑safe in CPython as stdout write is
        # protected by a lock; some interleaving is possible but acceptable).
        return _print_model_summary(curr_model_id, run_results)

    all_summaries: list[dict[str, Any]] = []

    if parallel and len(model_ids) > 1:
        # Run each model benchmark in its own worker thread.
        max_workers = min(len(model_ids), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(_run_model, mid): mid for mid in model_ids
            }

            for future in as_completed(future_to_model):
                mid = future_to_model[future]
                try:
                    per_size_stats = future.result()
                except Exception as exc:
                    click.echo(
                        click.style(
                            f"Unhandled exception for model {mid}: {exc}",
                            fg="red",
                            bold=True,
                        )
                    )
                    continue

                all_summaries.append({"model_id": mid, "size_summary": per_size_stats})
    else:
        # Sequential execution as before.
        for curr_model_id in model_ids:
            per_size_stats = _run_model(curr_model_id)
            all_summaries.append(
                {"model_id": curr_model_id, "size_summary": per_size_stats}
            )

    # ------- Cross‑model comparison table (only if >1 model) -------

    if len(all_summaries) > 1:
        click.echo("\n" + click.style("Comparison Table:", bold=True))
        headers = ["Model"] + [f"{w}x{h}" for w, h in sorted(sizes)]

        table_rows: list[list[str]] = [headers]
        for summ in all_summaries:
            row = [summ["model_id"]]
            for size in sorted(sizes):
                mat, cnt = summ["size_summary"].get(size, (0, 0))
                pct = mat / cnt * 100 if cnt else 0.0
                txt = f"{mat}/{cnt} ({pct:.0f}%)"
                if pct == 100:
                    fg = "green"
                elif pct >= 66.6667:
                    fg = "yellow"
                elif pct >= 33.3333:
                    fg = "bright_yellow"
                else:
                    fg = "red"
                row.append(click.style(txt, fg=fg))
            table_rows.append(row)

        def _strip_ansi(text: str) -> str:
            return ANSI_ESCAPE.sub("", text)

        col_widths = [
            max(len(_strip_ansi(r[i])) for r in table_rows) for i in range(len(headers))
        ]

        header_cells = [
            table_rows[0][i].ljust(col_widths[i]) for i in range(len(headers))
        ]
        click.echo("  ".join(header_cells))

        for row in table_rows[1:]:
            padded = [
                row[i] + " " * (col_widths[i] - len(_strip_ansi(row[i])))
                for i in range(len(headers))
            ]
            click.echo("  ".join(padded))

    # ------------------------------------------------------------------
    # Optional: visualisation
    # ------------------------------------------------------------------

    if plot:
        try:
            from ascii_maze_benchmark.visualization import plot_comparison_heatmap

            # Re‑use sizes list and all_summaries structure.
            plot_comparison_heatmap(
                all_summaries,
                sizes,
                output=plot_save,
                interactive=True,
            )
        except Exception as exc:  # pragma: no cover – plotting is best‑effort
            click.echo(
                click.style(f"Could not generate plot: {exc}", fg="red", bold=True)
            )

    return


if __name__ == "__main__":
    benchmark_command()
