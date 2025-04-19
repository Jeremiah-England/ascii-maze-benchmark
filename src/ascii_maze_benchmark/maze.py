import random
from collections import deque
from typing import List, Tuple, Deque, Set


def generate_maze(width: int, height: int, seed: int | None = None) -> List[str]:
    rng = random.Random(seed)
    if width < 1 or height < 1:
        raise ValueError("Maze width and height must be at least 1.")

    grid_width = 2 * width + 1
    grid_height = 2 * height + 1
    maze = [["#" for _ in range(grid_width)] for _ in range(grid_height)]
    visited = [[False for _ in range(width)] for _ in range(height)]
    stack: List[Tuple[int, int]] = []

    start_cell_row, start_cell_col = (
        rng.randint(0, height - 1),
        rng.randint(0, width - 1),
    )
    visited[start_cell_row][start_cell_col] = True
    stack.append((start_cell_row, start_cell_col))
    maze[2 * start_cell_row + 1][2 * start_cell_col + 1] = " "

    while stack:
        current_cell_row, current_cell_col = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            next_cell_row = current_cell_row + dr
            next_cell_col = current_cell_col + dc
            if (
                0 <= next_cell_row < height
                and 0 <= next_cell_col < width
                and not visited[next_cell_row][next_cell_col]
            ):
                neighbors.append(((next_cell_row, next_cell_col), (dr, dc)))

        if neighbors:
            next_cell_row, next_cell_col, dr, dc = rng.choice(neighbors)
            wall_row = 2 * current_cell_row + 1 + dr
            wall_col = 2 * current_cell_col + 1 + dc
            maze[wall_row][wall_col] = " "
            maze[2 * next_cell_row + 1][2 * next_cell_col + 1] = " "
            visited[next_cell_row][next_cell_col] = True
            stack.append((next_cell_row, next_cell_col))
        else:
            stack.pop()

    maze[0][1] = " "
    maze[grid_height - 1][grid_width - 2] = " "
    return ["".join(row) for row in maze]


def solve_maze(
    maze_list: list[str], return_raw_path: bool = False
) -> List[str] | tuple[List[str], List[Tuple[int, int]]]:
    height = len(maze_list)
    width = len(maze_list[0])
    maze = [list(row) for row in maze_list]
    start = None
    end = None

    for c in range(width):
        if maze[0][c] == " " and start is None:
            start = (0, c)
        if maze[height - 1][c] == " " and end is None:
            end = (height - 1, c)

    for r in range(1, height - 1):
        if maze[r][0] == " " and start is None:
            start = (r, 0)
        if maze[r][width - 1] == " " and end is None:
            end = (r, width - 1)

    if start is None or end is None:
        return maze_list if not return_raw_path else (maze_list, [])

    solution_path: List[Tuple[int, int]] | None = None
    queue: Deque[tuple[Tuple[int, int], List[Tuple[int, int]]]] = deque(
        [(start, [start])]
    )
    visited: Set[Tuple[int, int]] = set([start])

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            solution_path = path
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if _is_valid_move(nr, nc, height, width, maze, visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    if solution_path:
        solved_maze_list = [row[:] for row in maze]
        for r, c in solution_path:
            if solved_maze_list[r][c] == " ":
                solved_maze_list[r][c] = "."
        if return_raw_path:
            return ["".join(row) for row in solved_maze_list], solution_path
        return ["".join(row) for row in solved_maze_list]
    return maze_list if not return_raw_path else (maze_list, [])


def _is_valid_move(
    nr: int,
    nc: int,
    height: int,
    width: int,
    maze: List[List[str]],
    visited: Set[Tuple[int, int]],
) -> bool:
    return (
        0 <= nr < height
        and 0 <= nc < width
        and maze[nr][nc] != "#"
        and (nr, nc) not in visited
    )


def solution_to_directions(solution_path: List[Tuple[int, int]]) -> List[str]:
    if not solution_path or len(solution_path) < 2:
        return []
    directions = []
    for i in range(1, len(solution_path)):
        curr_r, curr_c = solution_path[i - 1]
        next_r, next_c = solution_path[i]
        if next_r < curr_r:
            directions.append("up")
        elif next_r > curr_r:
            directions.append("down")
        elif next_c < curr_c:
            directions.append("left")
        elif next_c > curr_c:
            directions.append("right")
    return directions
