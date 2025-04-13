import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import time

import click
import requests
from dotenv import load_dotenv

from ascii_maze_benchmark.generate_maze_script import generate_maze, solve_maze


class BenchmarkRunner:
    """Class to handle running the ASCII maze benchmark against various models."""

    def __init__(self, model_id: str, cache_dir: str = ".cache/benchmark_results"):
        """
        Initialize the benchmark runner.

        Args:
            model_id: The OpenRouter model ID to test
            cache_dir: Directory to cache results
        """
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
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache path for this specific model
        self.cache_path = self.cache_dir / f"{model_id.replace('/', '_')}.json"

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
        # Check for cached results
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                print(f"Loading cached results for {self.model_id}")
                return json.load(f)

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
                click.echo(
                    f"Testing maze {current_maze}/{total_mazes}: {width}x{height} (seed: {maze_seed})"
                )

                # Generate maze
                maze = generate_maze(width, height, maze_seed)
                if not maze:
                    click.echo(f"Failed to generate maze of size {width}x{height}")
                    continue

                # Solve maze to get the correct solution
                solution = solve_maze(maze)
                if not solution:
                    click.echo(f"Failed to solve maze of size {width}x{height}")
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
Only output the maze with the solution path marked.

Here's the maze:

{maze_str}

Provide only the maze with the solution path marked with '.' characters:
"""

                # Call the API to get the model's solution
                response = self._call_openrouter_api(prompt)
                model_solution = self._extract_solution(response)

                # Evaluate the solution
                exact_match = self._is_exact_match(solution, model_solution)
                levenshteins = self._calculate_levenshteins(solution, model_solution)

                # Store result
                result = {
                    "maze_size": (width, height),
                    "maze_seed": maze_seed,
                    "correct_solution": solution,
                    "model_solution": model_solution,
                    "exact_match": exact_match,
                    "levenshteins": levenshteins,
                }

                results["results"].append(result)

                # Save results after each maze to enable partial resume if needed
                with open(self.cache_path, "w") as f:
                    json.dump(results, f, indent=2)

        return results

    def _call_openrouter_api(self, prompt: str) -> Dict[str, Any]:
        """Call the OpenRouter API with the provided prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data
        )

        if response.status_code != 200:
            raise Exception(
                f"API call failed with status code {response.status_code}: {response.text}"
            )

        return response.json()

    def _extract_solution(self, response: Dict[str, Any]) -> List[str]:
        """Extract the solution from the API response."""
        try:
            # Extract the content from the assistant's message
            content = response["choices"][0]["message"]["content"]

            # Try to extract just the maze part using common patterns
            lines = content.strip().split("\n")

            # Filter out markdown code blocks if present
            if "```" in content:
                in_code_block = False
                maze_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block and line.strip() and ("#" in line or "." in line):
                        maze_lines.append(line)
                return maze_lines

            # Otherwise, look for lines containing maze characters
            maze_lines = [line for line in lines if ("#" in line or "." in line)]
            return maze_lines

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
    default=".cache/benchmark_results",
    help="Directory to cache benchmark results",
)
def benchmark_command(
    model_id: str, maze_sizes: str, mazes_per_size: int, seed: int, cache_dir: str
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
            "Error: Invalid maze sizes format. Use format: WIDTHxHEIGHT,WIDTHxHEIGHT"
        )
        return

    click.echo(f"Running benchmark on model: {model_id}")
    click.echo(f"Testing maze sizes: {maze_sizes}")
    click.echo(f"Mazes per size: {mazes_per_size}")

    try:
        runner = BenchmarkRunner(model_id, cache_dir)
        results = runner.run_benchmark(sizes, mazes_per_size, seed)

        # Analyze and print summary
        exact_matches = sum(1 for r in results["results"] if r["exact_match"])
        total = len(results["results"])

        click.echo("\nBenchmark Results Summary:")
        click.echo(f"Model: {model_id}")
        click.echo(
            f"Exact matches: {exact_matches}/{total} ({exact_matches / total * 100:.2f}%)"
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
            click.echo(f"{threshold}: {matches}/{total} ({matches / total * 100:.2f}%)")

        # Cache path for this model
        cache_path = Path(cache_dir) / f"{model_id.replace('/', '_')}.json"
        click.echo(f"\nDetailed results saved to: {cache_path}")

    except Exception as e:
        click.echo(f"Error running benchmark: {e}")


if __name__ == "__main__":
    benchmark_command()
