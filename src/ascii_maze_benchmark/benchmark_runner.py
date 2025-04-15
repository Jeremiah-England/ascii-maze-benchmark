import os
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import time

import click
import requests
from dotenv import load_dotenv
from diskcache import Cache

from ascii_maze_benchmark.generate_maze_script import generate_maze, solve_maze


class BenchmarkRunner:
    """Class to handle running the ASCII maze benchmark against various models."""

    def __init__(
        self,
        model_id: str,
        cache_dir: str = ".cache/api_responses",
        verbose: bool = False,
    ):
        """
        Initialize the benchmark runner.

        Args:
            model_id: The OpenRouter model ID to test
            cache_dir: Directory to cache API responses
            verbose: Whether to print detailed information during benchmarking
        """
        self.verbose = verbose
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
        # Set random seed for reproducibility
        random.seed(seed)

        results = {
            "model_id": self.model_id,
            "maze_sizes": maze_sizes,
            "num_mazes_per_size": num_mazes_per_size,
            "seed": seed,
            "timestamp": time.time(),
            "results": [],
        }

        maze_seeds = [random.randint(1, 10000) for _ in range(num_mazes_per_size)]

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

                # Solve maze to get the correct solution
                solution = solve_maze(maze)
                if not solution:
                    click.echo(
                        click.style(
                            f"Failed to solve maze of size {width}x{height}", fg="red"
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

                # Create prompt
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

                # Evaluate the solution
                exact_match = self._is_exact_match(solution, model_solution)
                levenshteins = self._calculate_levenshteins(solution, model_solution)

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

                # Display Levenshtein distance
                match_text = "✓" if exact_match else "✗"
                click.echo(
                    f"Solution: {click.style(match_text, fg='green' if exact_match else 'red', bold=True)} (Levenshtein distance: {click.style(str(distance), fg='cyan', bold=True)})"
                )

                # Store result
                result = {
                    "maze_size": (width, height),
                    "maze_seed": maze_seed,
                    "correct_solution": solution,
                    "model_solution": model_solution,
                    "exact_match": exact_match,
                    "levenshteins": levenshteins,
                    "levenshtein_distance": distance,
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

    def _extract_solution(self, response: Dict[str, Any]) -> List[str]:
        """Extract the solution from the API response."""
        try:
            # Extract the content from the assistant's message
            content = response["choices"][0]["message"]["content"]

            if self.verbose:
                click.echo("\n=== Model Response ===")
                click.echo(click.style(content, fg="bright_black"))
                click.echo("=====================\n")

            return self.extract_solution_from_content(content)

        except (KeyError, IndexError):
            return []

    def _is_exact_match(
        self, correct_solution: List[str], model_solution: List[str]
    ) -> bool:
        """Check if the model's solution exactly matches the correct solution."""
        if len(correct_solution) != len(model_solution):
            return False

        for i in range(len(correct_solution)):
            if correct_solution[i] != model_solution[i]:
                return False

        return True

    def _calculate_levenshteins(
        self, correct_solution: List[str], model_solution: List[str]
    ) -> Dict[str, bool]:
        """Calculate Levenshtein distances for different thresholds."""
        # Join all lines to create a single string for comparison
        correct_str = "\n".join(correct_solution)
        model_str = "\n".join(model_solution)

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
@click.argument("model_id", type=str)
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
def benchmark_command(
    model_id: str,
    maze_sizes: str,
    mazes_per_size: int,
    seed: int,
    cache_dir: str,
    verbose: bool = False,
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

    click.echo(f"Running benchmark on model: {click.style(model_id, fg='bright_blue')}")
    click.echo(f"Testing maze sizes: {click.style(maze_sizes, fg='cyan')}")
    click.echo(f"Mazes per size: {click.style(str(mazes_per_size), fg='cyan')}")
    if verbose:
        click.echo(f"Verbose mode: {click.style('ON', fg='green', bold=True)}")

    try:
        runner = BenchmarkRunner(model_id, cache_dir, verbose)
        results = runner.run_benchmark(sizes, mazes_per_size, seed)

        # Analyze and print summary
        exact_matches = sum(1 for r in results["results"] if r["exact_match"])
        total = len(results["results"])

        click.echo("\n" + click.style("Benchmark Results Summary:", bold=True))
        click.echo(f"Model: {click.style(model_id, fg='bright_blue', bold=True)}")
        percentage = exact_matches / total * 100
        color = "green" if percentage >= 75 else "yellow" if percentage >= 50 else "red"

        # Calculate average Levenshtein distance
        avg_distance = (
            sum(r["levenshtein_distance"] for r in results["results"]) / total
        )
        click.echo(
            f"Exact matches: {click.style(f'{exact_matches}/{total} ({percentage:.2f}%)', fg=color, bold=True)} (Avg Levenshtein distance: {click.style(f'{avg_distance:.2f}', fg='cyan', bold=True)})"
        )

        # Compute average scores for different Levenshtein thresholds
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
            matches = sum(1 for r in results["results"] if r["levenshteins"][threshold])
            match_percentage = matches / total * 100
            threshold_color = (
                "green"
                if match_percentage >= 75
                else "yellow"
                if match_percentage >= 50
                else "red"
            )
            # Format the threshold name nicely
            threshold_display = (
                threshold.replace("_", " ").replace("percent", "within").title()
            )
            click.echo(
                f"{threshold_display}: {click.style(f'{matches}/{total} ({match_percentage:.2f}%)', fg=threshold_color)}"
            )

        # No results file is saved - just a summary to the console

    except Exception as e:
        click.echo(click.style(f"Error running benchmark: {e}", fg="red", bold=True))


if __name__ == "__main__":
    benchmark_command()
