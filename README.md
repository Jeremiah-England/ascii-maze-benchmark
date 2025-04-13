# Ascii Maze Benchmark

This is a benchmark for testing how capable different LLMs are at solving ascii mazes.
Here is an example 4x4 maze:

```
START
 v
# #######
#       #
# ##### #
# #     #
# #######
#     # #
##### # #
#       #
####### #
       ^
   FINISH
```

Here is the solution:

```
#.#######
#.      #
#.##### #
#.#     #
#.#######
#.....# #
#####.# #
#    ...#
#######.#
```

The benchmark randomly generates mazes from a seed, and evaluates LLMs ability to solve the maze.

Some LLMs tend to struggle with perfectly formatting the output for some reason, so we report scores at varying string distances to the correct response.

We evaluate all models using the OpenRouter API, to keep it simple. If it's not on open router, the benchmark will not be run.

## Usage

### Setup

1. Copy `.env.example` to `.env` and add your OpenRouter API key:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file and replace `your_api_key_here` with your actual OpenRouter API key.

### Generate Example Mazes

To generate and solve an example maze:

```
uv run ascii-maze-benchmark generate-example WIDTH HEIGHT [--seed SEED]
```

Example:
```
uv run ascii-maze-benchmark generate-example 5 5 --seed 42
```

### Run Benchmarks

To run benchmarks against a specific model:

```
uv run ascii-maze-benchmark run-benchmark MODEL_ID [OPTIONS]
```

Options:
- `--maze-sizes TEXT`: Comma-separated list of maze sizes to test (format: WIDTHxHEIGHT)
- `--mazes-per-size INTEGER`: Number of different mazes to generate per size
- `--seed INTEGER`: Random seed for reproducible maze generation
- `--cache-dir TEXT`: Directory to cache benchmark results

Example:
```
uv run ascii-maze-benchmark run-benchmark anthropic/claude-3-haiku-20240307 --maze-sizes 3x3,4x4 --mazes-per-size 2
```

## Development Tips

- Benchmark results are cached in the `.cache/benchmark_results` directory by default, so visualization code can be rerun without spending money to rerun the benchmark.
- Test the benchmarking code on a cheap model on OpenRouter first, to save costs.
- Use the `.env` file to manage OpenRouter credentials.
- Use `uv` for package management and running commands.
- There is a `src/ascii_maze_benchmark/generate_maze_script.py` file you can use as a reference for maze generation logic.
