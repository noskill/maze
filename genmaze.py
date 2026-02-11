import argparse
import random


DIRS = (
    (0, 1, 1, 4),   # up
    (1, 0, 2, 8),   # right
    (0, -1, 4, 1),  # down
    (-1, 0, 8, 2),  # left
)


def generate_maze(dim, seed=None, extra_openings=0):
    rng = random.Random(seed)
    walls = [[0 for _ in range(dim)] for _ in range(dim)]
    visited = [[False for _ in range(dim)] for _ in range(dim)]

    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy, bit, opp in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim and 0 <= ny < dim and not visited[nx][ny]:
                neighbors.append((nx, ny, bit, opp))
        if not neighbors:
            stack.pop()
            continue

        nx, ny, bit, opp = rng.choice(neighbors)
        walls[x][y] |= bit
        walls[nx][ny] |= opp
        visited[nx][ny] = True
        stack.append((nx, ny))

    for _ in range(extra_openings):
        x = rng.randrange(dim)
        y = rng.randrange(dim)
        dx, dy, bit, opp = rng.choice(DIRS)
        nx, ny = x + dx, y + dy
        if 0 <= nx < dim and 0 <= ny < dim:
            walls[x][y] |= bit
            walls[nx][ny] |= opp

    return walls


def write_maze(path, dim, walls):
    with open(path, "w") as f_out:
        f_out.write(str(dim) + "\n")
        for x in range(dim):
            row = ",".join(str(walls[x][y]) for y in range(dim))
            f_out.write(row + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a random maze.")
    parser.add_argument("dim", type=int, help="Even maze size (e.g. 12).")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output filename (default: random_maze_<dim>.txt).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--extra-openings",
        type=int,
        default=0,
        help="Number of additional openings to add loops.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dim = args.dim
    if dim % 2:
        raise SystemExit("Maze dimensions must be even.")
    if dim <= 1:
        raise SystemExit("Maze dimensions must be > 1.")

    output = args.output or "random_maze_{}.txt".format(dim)
    walls = generate_maze(dim, seed=args.seed, extra_openings=args.extra_openings)
    write_maze(output, dim, walls)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
