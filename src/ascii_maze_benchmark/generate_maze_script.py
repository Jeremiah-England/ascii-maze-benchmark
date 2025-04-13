import random
import sys
from collections import deque

import click


def generate_maze(width: int, height: int, seed: int | None = None):
    """
    Generates an ASCII maze of specified width and height with an entrance and exit.

    Args:
        width (int): The number of cells wide the maze should be.
        height (int): The number of cells high the maze should be.
        seed (int, optional): Random seed for reproducible maze generation. Defaults to None.

    Returns
    -------
        list: A list of strings representing the ASCII maze.
              Returns None if width or height is less than 1.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    if width < 1 or height < 1:
        print("Error: Maze width and height must be at least 1.")
        return None

    # Maze grid dimensions (including walls)
    grid_width = 2 * width + 1
    grid_height = 2 * height + 1

    # Initialize grid with walls ('#')
    maze = [["#" for _ in range(grid_width)] for _ in range(grid_height)]

    # Keep track of visited cells (using cell coordinates, not grid coordinates)
    visited = [[False for _ in range(width)] for _ in range(height)]

    # Stack for DFS backtracking (stores (row, col) of cells)
    stack = []

    # --- Maze Generation Algorithm (Recursive Backtracker / DFS) ---

    # 1. Choose a random starting cell (doesn't have to be the entrance)
    start_cell_row, start_cell_col = (
        random.randint(0, height - 1),
        random.randint(0, width - 1),
    )
    visited[start_cell_row][start_cell_col] = True
    stack.append((start_cell_row, start_cell_col))

    # Convert cell coordinates to grid coordinates for path carving
    # Cell (r, c) corresponds to grid position (2*r + 1, 2*c + 1)
    maze[2 * start_cell_row + 1][2 * start_cell_col + 1] = " "

    while stack:
        current_cell_row, current_cell_col = stack[-1]

        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:  # N, S, E, W
            next_cell_row, next_cell_col = current_cell_row + dr, current_cell_col + dc

            if 0 <= next_cell_row < height and 0 <= next_cell_col < width:
                if not visited[next_cell_row][next_cell_col]:
                    neighbors.append(((next_cell_row, next_cell_col), (dr, dc)))

        if neighbors:
            (next_cell_row, next_cell_col), (dr, dc) = random.choice(neighbors)

            # Remove the wall between cells
            wall_row = 2 * current_cell_row + 1 + dr
            wall_col = 2 * current_cell_col + 1 + dc
            maze[wall_row][wall_col] = " "

            # Carve path in the neighbor cell
            maze[2 * next_cell_row + 1][2 * next_cell_col + 1] = " "

            visited[next_cell_row][next_cell_col] = True
            stack.append((next_cell_row, next_cell_col))
        else:
            stack.pop()  # Backtrack

    # --- Create Entrance and Exit ---
    # Break the outer wall at specific points.
    # Ensure these points connect to a path cell (odd indices usually work best).

    # Entrance: Top wall, near the first column's path space
    # maze[0] is the top row.
    # maze[0][1] corresponds to the space above the first cell (0,0)'s path at (1,1).
    maze[0][1] = " "

    # Exit: Bottom wall, near the last column's path space
    # maze[grid_height - 1] is the bottom row.
    # maze[grid_height - 1][grid_width - 2] corresponds to the space below
    # the last cell (height-1, width-1)'s path at (grid_height-2, grid_width-2).
    maze[grid_height - 1][grid_width - 2] = " "

    # --- Optional: Alternative Entrance/Exit Placement ---
    # You could place them elsewhere, for example:
    # maze[1][0] = ' ' # Left wall, near the top
    # maze[grid_height - 2][grid_width - 1] = ' ' # Right wall, near the bottom
    # Make sure the chosen coordinates break an outer '#' wall
    # adjacent to an inner ' ' path space.

    # Convert the grid (list of lists) into a list of strings
    return ["".join(row) for row in maze]


def print_maze(maze_list: list[str]):
    """Prints the maze list to the console."""
    print("START")
    print(r" v")
    for row in maze_list:
        print(row)
    print(" " * (len(maze_list[0]) - 2) + "^")
    print(" " * (len(maze_list[0]) - 6) + "FINISH")


def solve_maze(maze_list: list[str]):
    """
    Solves an ASCII maze represented by a list of strings.

    Args:
        maze_list (list): A list of strings representing the maze.
                          '#' = wall, ' ' = path.

    Returns
    -------
        list: A list of strings representing the maze with the solution path
              marked by '.', or None if no solution is found or maze is invalid.
              Returns the original list if start/end not found.
    """
    height = len(maze_list)
    width = len(maze_list[0])
    maze = [list(row) for row in maze_list]  # Convert to list of lists for mutability

    start = None
    end = None

    # 1. Find Start and End points (spaces on the border)
    # Check top/bottom borders
    for c in range(width):
        if maze[0][c] == " ":
            if start is None:
                start = (0, c)
            else:
                end = (0, c)
        if maze[height - 1][c] == " ":
            if start is None:
                start = (height - 1, c)
            else:
                end = (height - 1, c)
    # Check left/right borders (avoid double-checking corners)
    for r in range(1, height - 1):
        if maze[r][0] == " ":
            if start is None:
                start = (r, 0)
            else:
                end = (r, 0)
        if maze[r][width - 1] == " ":
            if start is None:
                start = (r, width - 1)
            else:
                end = (r, width - 1)

    if start is None or end is None:
        print(
            "Error: Could not find exactly two openings (start and end) on the border."
        )
        # Return the original maze maybe? Or indicate error.
        return ["".join(row) for row in maze]  # Return original if no start/end

    # 2. BFS Implementation
    queue = deque([(start, [start])])  # Store (current_position, path_so_far)
    visited = {start}

    solution_path = None

    while queue:
        (r, c), path = queue.popleft()

        # Check if we reached the end
        if (r, c) == end:
            solution_path = path
            break

        # Explore neighbors (Up, Down, Left, Right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Check if neighbor is valid:
            # 1. Within bounds
            # 2. Not a wall
            # 3. Not visited yet
            if (
                0 <= nr < height
                and 0 <= nc < width
                and maze[nr][nc] != "#"
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                new_path = path + [(nr, nc)]  # Create the new path
                queue.append(((nr, nc), new_path))

    # 3. Mark the solution path on the maze
    if solution_path:
        solved_maze_list = [row[:] for row in maze]  # Create a copy to modify
        for r, c in solution_path:
            # Avoid overwriting start/end if they are walls technically
            # (though they should be spaces found earlier)
            if solved_maze_list[r][c] == " ":
                solved_maze_list[r][c] = "."  # Mark path with '.'
        # Optional: Mark Start and End differently
        # sr, sc = start
        # er, ec = end
        # solved_maze_list[sr][sc] = 'S'
        # solved_maze_list[er][ec] = 'E'
        return ["".join(row) for row in solved_maze_list]

    return None


@click.command(name="generate-example")
@click.argument("width", type=int)
@click.argument("height", type=int)
@click.option("--seed", type=int, help="Random seed for reproducible maze generation")
def generate_maze_command(width, height, seed):
    """Generate an ASCII maze with entrance and exit."""
    print(f"\nGenerating a {width}x{height} maze...\n")
    generated_maze = generate_maze(width, height, seed)

    if not generated_maze:
        sys.exit("No such maze")

    print_maze(generated_maze)

    if generated_maze:
        print("\nSolving maze...\n")

        solution = solve_maze(generated_maze)
        if solution is None:
            print("Could not solve maze.")
        else:
            print("\n".join(solution))


# For backwards compatibility with direct script execution
if __name__ == "__main__":
    generate_maze_command()
