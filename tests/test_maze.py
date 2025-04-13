import pytest

from ascii_maze_benchmark.generate_maze_script import generate_maze, solve_maze


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
    maze = generate_maze(width, height, 42)  # Use consistent seed

    # Check that the result is None for invalid inputs
    assert maze is None


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
