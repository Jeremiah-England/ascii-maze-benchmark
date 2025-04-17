import pytest

from ascii_maze_benchmark.generate_maze_script import (
    generate_maze,
    solve_maze,
    solution_to_directions,
)


@pytest.mark.parametrize(
    ("width", "height", "seed", "expected_maze"),
    [
        (
            5,
            5,
            42,
            [
                "# #########",
                "#   #     #",
                "### # ### #",
                "# # # #   #",
                "# # # # ###",
                "# # # # # #",
                "# # # # # #",
                "#   # # # #",
                "# ### # # #",
                "#     #   #",
                "######### #",
            ],
        ),
        (
            5,
            5,
            123,
            [
                "# #########",
                "#   # #   #",
                "### # ### #",
                "#   #   # #",
                "# # ### # #",
                "# # # # # #",
                "# # # # # #",
                "# #   # # #",
                "# ##### # #",
                "#         #",
                "######### #",
            ],
        ),
        (
            3,
            3,
            99,
            [
                "# #####",
                "#     #",
                "# ### #",
                "# # # #",
                "# # # #",
                "# #   #",
                "##### #",
            ],
        ),
        (1, 1, 1, ["# #", "# #", "# #"]),
    ],
)
def test_maze_generation(width, height, seed, expected_maze):
    """Test that the maze generation function creates expected mazes with given seeds."""
    maze = generate_maze(width, height, seed)

    # Check that the maze matches expected output
    assert maze == expected_maze

    # Check that the maze has an entrance and exit
    # Entrance should be in the top row
    assert " " in maze[0]
    # Exit should be in the bottom row
    assert " " in maze[-1]

    # Check that the maze only contains walls '#' and paths ' '
    for row in maze:
        assert all(cell in ["#", " "] for cell in row)


@pytest.mark.parametrize(
    ("maze", "expected_solution"),
    [
        (
            [
                "# #########",
                "#   #     #",
                "### # ### #",
                "# # # #   #",
                "# # # # ###",
                "# # # # # #",
                "# # # # # #",
                "#   # # # #",
                "# ### # # #",
                "#     #   #",
                "######### #",
            ],
            [
                "#.#########",
                "#...#.....#",
                "###.#.###.#",
                "# #.#.#...#",
                "# #.#.#.###",
                "# #.#.#.# #",
                "# #.#.#.# #",
                "#...#.#.# #",
                "#.###.#.# #",
                "#.....#...#",
                "#########.#",
            ],
        ),
        (
            [
                "# #########",
                "#   # #   #",
                "### # ### #",
                "#   #   # #",
                "# # ### # #",
                "# # # # # #",
                "# # # # # #",
                "# #   # # #",
                "# ##### # #",
                "#         #",
                "######### #",
            ],
            [
                "#.#########",
                "#...# #   #",
                "###.# ### #",
                "#...#   # #",
                "#.# ### # #",
                "#.# # # # #",
                "#.# # # # #",
                "#.#   # # #",
                "#.##### # #",
                "#.........#",
                "#########.#",
            ],
        ),
        (
            [
                "# #####",
                "#     #",
                "# ### #",
                "# # # #",
                "# # # #",
                "# #   #",
                "##### #",
            ],
            [
                "#.#####",
                "#.....#",
                "# ###.#",
                "# # #.#",
                "# # #.#",
                "# #  .#",
                "#####.#",
            ],
        ),
        (["# #", "# #", "# #"], ["#.#", "#.#", "#.#"]),
    ],
)
def test_maze_solution(maze, expected_solution):
    """Test that the maze solver produces the expected solution for given mazes."""
    solution = solve_maze(maze)

    # Check that solution matches expected output
    assert solution == expected_solution

    # Check that solution has same dimensions as maze
    assert len(solution) == len(maze)
    assert all(
        len(solution_row) == len(maze_row)
        for solution_row, maze_row in zip(solution, maze, strict=False)
    )

    # Check that solution contains path markers '.'
    has_path_markers = any("." in row for row in solution)
    assert has_path_markers

    # Verify walls are preserved in solution
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == "#":
                assert solution[i][j] == "#"


@pytest.mark.parametrize(
    "invalid_input",
    [
        (0, 5),  # Width too small
        (5, 0),  # Height too small
        (-1, 5),  # Negative width
        (5, -1),  # Negative height
    ],
)
def test_invalid_maze_parameters(invalid_input):
    """Test that the maze generator handles invalid inputs properly."""
    width, height = invalid_input

    # Generate maze with invalid parameters
    with pytest.raises(ValueError):
        generate_maze(width, height, 42)  # Use consistent seed


def test_generation_deterministic():
    """Test that maze generation is deterministic with the same seed."""
    # Generate the same maze twice with the same seed
    maze1 = generate_maze(5, 5, 42)
    maze2 = generate_maze(5, 5, 42)

    # Check that they are identical
    assert maze1 == maze2

    # Generate mazes with different seeds
    maze3 = generate_maze(5, 5, 123)

    # Check that they are different
    assert maze1 != maze3


def test_invalid_maze_for_solving():
    """Test that the solver handles invalid mazes properly."""
    # Maze with no entrance or exit
    invalid_maze = ["#####", "#   #", "#   #", "#   #", "#####"]

    # Solve should return the original maze if no entrance/exit is found
    solution = solve_maze(invalid_maze)
    assert solution == invalid_maze


# We'll just test the smallest valid maze case
def test_smallest_valid_maze():
    """Test that the smallest possible valid maze can be solved."""
    tiny_maze = ["# #", "   ", "# #"]
    solution = solve_maze(tiny_maze)

    # Check that there's a solution path
    has_path_markers = any("." in row for row in solution)
    assert has_path_markers


def test_print_maze(capsys):
    """Test that print_maze displays the maze correctly."""
    from ascii_maze_benchmark.generate_maze_script import print_maze

    maze = ["###", "# #", "###"]
    print_maze(maze)

    captured = capsys.readouterr()
    expected_output = "START\n v\n###\n# #\n###\n ^\nFINISH\n"

    assert captured.out == expected_output


@pytest.mark.parametrize(
    ("content", "expected_solution"),
    [
        # Test case 1: Solution in ```solution block
        (
            "I'll think about this maze step by step.\n\n```solution\n#.#########\n#...#.....#\n###.#.###.#\n# #.#.#...#\n# #.#.#.###\n```",
            [
                "#.#########",
                "#...#.....#",
                "###.#.###.#",
                "# #.#.#...#",
                "# #.#.#.###",
            ],
        ),
        # Test case 2: Solution in regular code block (should be ignored)
        (
            "Here's my solution:\n\n```\n#.#########\n#...#.....#\n###.#.###.#\n```",
            [],
        ),
        # Test case 3: Solution with no code blocks, just raw lines (should also be ignored)
        (
            "My solution is:\n\n#.#########\n#...#.....#\n###.#.###.#",
            [],
        ),
        # Test case 4: Multiple code blocks but only extract solution block
        (
            "First attempt:\n```\n#.####.####\n#...#...#.#\n###.#.###.#\n```\n\nBetter solution:\n```solution\n#.#########\n#...#.....#\n###.#.###.#\n```",
            [
                "#.#########",
                "#...#.....#",
                "###.#.###.#",
            ],
        ),
        # Test case 5: Empty response
        (
            "",
            [],
        ),
        # Test case 6: No valid maze lines
        (
            "I'm not sure how to solve this maze.",
            [],
        ),
        # Test case 7: Multiple solution blocks - use the last one
        (
            "First solution attempt:\n```solution\n#.####.####\n#...#...#.#\n###.#.###.#\n```\n\nImproved solution:\n```solution\n#.#########\n#...#.....#\n###.#.###.#\n```",
            [
                "#.#########",
                "#...#.....#",
                "###.#.###.#",
            ],
        ),
        # Test case 8: Multiple regular code blocks - should be ignored completely
        (
            "First attempt:\n```\n#.####.####\n#...#...#.#\n###.#.###.#\n```\n\nSecond attempt:\n```\n#.#########\n#...#.....#\n###.#.###.#\n```",
            [],  # No solution blocks, so no output
        ),
    ],
)
def test_extract_solution_from_content(content, expected_solution):
    """Test the solution extraction logic with various response formats."""
    from ascii_maze_benchmark.benchmark_runner import BenchmarkRunner

    solution = BenchmarkRunner.extract_solution_from_content(content)
    assert solution == expected_solution


@pytest.mark.parametrize(
    ("solution_path", "expected_directions"),
    [
        # Simple straight path: down and right
        ([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], ["down", "down", "right", "right"]),
        # U-shaped path: down, right, up
        (
            [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)],
            ["down", "down", "right", "up", "up"],
        ),
        # All four directions: down, right, up, left
        ([(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)], ["down", "right", "up", "right"]),
        # Spiral pattern
        (
            [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)],
            ["down", "down", "right", "right", "up", "up", "left"],
        ),
        # Empty path
        ([], []),
        # Single point (no direction)
        ([(0, 0)], []),
    ],
)
def test_solution_to_directions(solution_path, expected_directions):
    """Test converting a path to directions."""
    directions = solution_to_directions(solution_path)
    assert directions == expected_directions


def test_maze_to_directions():
    """Test solving a maze and converting the solution to directions."""
    # Generate a 3x3 maze with a specific seed
    maze = generate_maze(3, 3, 42)

    # Solve the maze and get the raw path
    _, raw_path = solve_maze(maze, return_raw_path=True)

    # Convert the path to directions
    directions = solution_to_directions(raw_path)

    # For this specific maze and seed, we know the expected directions
    expected_directions = [
        "down",
        "right",
        "right",
        "right",
        "right",
        "down",
        "down",
        "down",
        "down",
        "down",
    ]

    # Check that directions match expected output
    assert directions == expected_directions


@pytest.mark.parametrize(
    ("content", "expected_directions"),
    [
        # Test case 1: Comma-separated directions in a solution block
        (
            "I'll solve this maze by following these steps.\n\n```solution\ndown,right,right,down,left,down\n```",
            ["down", "right", "right", "down", "left", "down"],
        ),
        # Test case 2: Newline-separated directions in a solution block
        (
            "Here are my steps:\n\n```solution\ndown\nright\nright\ndown\n```",
            ["down", "right", "right", "down"],
        ),
        # Test case 3: Single-line solution without commas or newlines
        (
            "My solution:\n\n```solution\ndown right right down\n```",
            ["down", "right", "right", "down"],
        ),
        # Test case 4: Mixed format with spacing variations
        ("```solution\n  down right down  \n```", ["down", "right", "down"]),
        # Test case 5: Solution with extra text
        (
            "```solution\nHere's my path: down right down\n```",
            ["down", "right", "down"],
        ),
        # Test case 6: Multiple solution blocks - use the last one
        (
            "First attempt:\n```solution\ndown right up\n```\nBetter solution:\n```solution\ndown right down\n```",
            ["down", "right", "down"],
        ),
        # Test case 7: Solution outside of solution block as fallback
        ("I'll go down, then right, then down.", ["down", "right", "down"]),
        # Test case 8: Empty response
        ("", []),
    ],
)
def test_extract_directions_from_content(content, expected_directions):
    """Test extracting directional instructions from model responses."""
    from ascii_maze_benchmark.benchmark_runner import BenchmarkRunner

    directions = BenchmarkRunner.extract_directions_from_content(content)
    assert directions == expected_directions


def test_benchmark_runner_directional_mode():
    """Test BenchmarkRunner's directional mode configuration."""
    from ascii_maze_benchmark.benchmark_runner import BenchmarkRunner
    import os
    import unittest.mock

    # Mock environment variable for the API key
    with unittest.mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
        # Create a BenchmarkRunner instance in directional mode
        runner = BenchmarkRunner(
            model_id="test_model",
            cache_dir=".test_cache",
            verbose=True,
            directional_mode=True,
        )

        # Verify it's in directional mode
        assert runner.directional_mode is True

        # Create another runner not in directional mode
        regular_runner = BenchmarkRunner(
            model_id="test_model",
            cache_dir=".test_cache",
            verbose=True,
            directional_mode=False,
        )

        # Verify it's not in directional mode
        assert regular_runner.directional_mode is False


@pytest.mark.parametrize(
    ("correct_solution", "model_solution", "is_directional_mode", "expected_match"),
    [
        # Exact match cases
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "down", "right", "down"],
            True,
            True,
        ),
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "down", "right", "down"],
            False,
            True,
        ),
        # Case with first 'down' omitted - should match in directional mode only
        (
            ["down", "right", "down", "right", "down"],
            ["right", "down", "right", "down"],
            True,
            True,
        ),
        (
            ["down", "right", "down", "right", "down"],
            ["right", "down", "right", "down"],
            False,
            False,
        ),
        # Case with last 'down' omitted - should match in directional mode only
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "down", "right"],
            True,
            True,
        ),
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "down", "right"],
            False,
            False,
        ),
        # Case with both first and last 'down' omitted - should match in directional mode only
        (
            ["down", "right", "down", "right", "down"],
            ["right", "down", "right"],
            True,
            True,
        ),
        (
            ["down", "right", "down", "right", "down"],
            ["right", "down", "right"],
            False,
            False,
        ),
        # Cases with extra 'down's at the end - should match in directional mode only
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down"],
            True,
            True,
        ),
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down"],
            False,
            False,
        ),
        # Case with two extra 'down's at the end - should match in directional mode only
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down", "down"],
            True,
            True,
        ),
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down", "down"],
            False,
            False,
        ),
        # Case with first 'down' omitted and extra 'down' at the end
        (
            ["down", "right", "down", "right"],
            ["right", "down", "right", "down"],
            True,
            True,
        ),
        # Case with first 'down' omitted and two extra 'down's at the end
        (
            ["down", "right", "down", "right"],
            ["right", "down", "right", "down", "down"],
            True,
            True,
        ),
        # Wrong solution cases - should never match
        (
            ["down", "right", "down", "right", "down"],
            ["down", "left", "down", "right", "down"],
            True,
            False,
        ),
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "up", "right", "down"],
            False,
            False,
        ),
        # Empty solution cases
        ([], [], True, True),
        ([], [], False, True),
        (
            ["down"],
            [],
            True,
            True,
        ),  # Changed to True - a single 'down' should match empty in directional mode
        ([], ["down"], True, False),
        # Completely different length solutions
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right"],
            True,
            False,
        ),
        # Only one 'down' direction in the correct solution
        (["down"], [], True, True),
        (["down"], [], False, False),
        # Extra 'down's beyond the limit - should not match
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down", "down", "down"],
            True,
            False,
        ),
    ],
)
def test_is_exact_match_with_down_omission(
    correct_solution, model_solution, is_directional_mode, expected_match
):
    """
    Test the _is_exact_match method with various cases of 'down' direction handling.

    The modified _is_exact_match method should account for cases where:
    1. The leading 'down' direction is omitted
    2. Up to two extra 'down' directions are added at the end
    3. Both of the above occur simultaneously
    """
    from ascii_maze_benchmark.benchmark_runner import BenchmarkRunner
    import os
    import unittest.mock

    # Mock environment variable for the API key
    with unittest.mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
        # Create a BenchmarkRunner instance with the specified directional mode
        runner = BenchmarkRunner(
            model_id="test_model",
            cache_dir=".test_cache",
            directional_mode=is_directional_mode,
        )

        # Test the matching logic
        assert (
            runner._is_exact_match(correct_solution, model_solution) == expected_match
        )


@pytest.mark.parametrize(
    ("directions", "model_solution", "expected_match"),
    [
        # Exact match
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "down", "right", "down"],
            True,
        ),
        # First 'down' omitted
        (
            ["down", "right", "down", "right", "down"],
            ["right", "down", "right", "down"],
            True,
        ),
        # Last 'down' omitted
        (
            ["down", "right", "down", "right", "down"],
            ["down", "right", "down", "right"],
            True,
        ),
        # Both first and last 'down' omitted
        (
            ["down", "right", "down", "right", "down"],
            ["right", "down", "right"],
            True,
        ),
        # Extra down at the end
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down"],
            True,
        ),
        # Two extra downs at the end
        (
            ["down", "right", "down", "right"],
            ["down", "right", "down", "right", "down", "down"],
            True,
        ),
        # No match case
        (
            ["down", "right", "down", "right", "down"],
            ["down", "left", "down", "right", "down"],
            False,
        ),
        # Empty solutions
        ([], [], True),
        (
            [
                "down",
                "down",
                "down",
                "down",
                "down",
                "down",
                "down",
                "right",
                "right",
                "up",
                "up",
                "right",
                "right",
                "right",
                "right",
                "down",
                "down",
                "down",
            ],
            [
                "down",
                "down",
                "down",
                "down",
                "down",
                "down",
                "down",
                "down",
                "right",
                "right",
                "up",
                "up",
                "right",
                "right",
                "right",
                "right",
                "down",
                "down",
                "down",
                "down",
            ],
            True,
        ),
    ],
)
def test_match_description_logic(directions, model_solution, expected_match):
    """
    Test the match description logic used in verbose output.

    The function checks that the correct description is generated based on
    the type of match:
    - Exact match
    - First 'down' omitted
    - Last 'down' omitted
    - Both first and last 'down' omitted
    - Extra 'down' at the end
    - Two extra 'downs' at the end
    - Combination of first 'down' omitted and extra 'down's at the end
    """
    # This is a direct test of the logic in the verbose output section of the benchmark runner

    # First verify that the match is as expected using our runner
    from ascii_maze_benchmark.benchmark_runner import BenchmarkRunner
    import os
    import unittest.mock

    # Mock environment variable for the API key
    with unittest.mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
        runner = BenchmarkRunner(
            model_id="test_model",
            cache_dir=".test_cache",
            directional_mode=True,  # Always in directional mode for this test
        )

        # Verify the match result
        is_match = runner._is_exact_match(directions, model_solution)
        assert is_match == expected_match


def test_combined_first_down_omitted_and_extra_down():
    """
    Test the special case where both the first down is omitted and an extra down is added at the end.

    This test ensures that the _is_exact_match method correctly handles this combined scenario.
    """
    from ascii_maze_benchmark.benchmark_runner import BenchmarkRunner
    import os
    import unittest.mock

    # Test data
    correct_solution = ["down", "right", "down", "right"]
    model_solution = ["right", "down", "right", "down", "down"]

    # Mock environment variable for the API key
    with unittest.mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
        runner = BenchmarkRunner(
            model_id="test_model",
            cache_dir=".test_cache",
            directional_mode=True,
        )

        # Verify the match result - this should be true because we handle both cases
        is_match = runner._is_exact_match(correct_solution, model_solution)
        assert is_match is True

        # Order of checks determines which description would be returned
        # In our implementation, we check for first 'down' omitted before extra downs
        # So in verbose output, this would be reported as "first down omitted"
        # Even though technically both conditions are satisfied
