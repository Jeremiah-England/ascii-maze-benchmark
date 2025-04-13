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

## Development Tips

- Cache benchmark results for different models so any visualization code can be rerun without spending money to rerun the benchmark.
- Test the benchmarking code on a really really cheap model on OpenRouter, to save costs.
- Use a .env file to manage OpenRouter credentials.
- See the initial maze printing and solving logic in
- Use `uv` if we need to install dependencies for anything.
- There is a `src/ascii_maze_benchmark/generate_maze_script.py` file you can use as a reference
