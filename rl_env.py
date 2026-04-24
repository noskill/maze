"""Vectorized maze environment for policy/world-model training.

Design goals:
- Keep dynamics aligned with `tester.py` semantics.
- Support vectorized headless rollouts for training.
- Support optional single-env visualization with existing pygame image code.
- Keep observation contract simple and stable.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from genmaze import generate_maze, write_maze
from maze import Maze


HEADINGS: Tuple[str, ...] = ("up", "right", "down", "left")
HEADING_TO_IDX = {name: idx for idx, name in enumerate(HEADINGS)}

DIR_SENSORS = {
    "up": ("left", "up", "right"),
    "right": ("up", "right", "down"),
    "down": ("right", "down", "left"),
    "left": ("down", "left", "up"),
}

DIR_MOVE = {
    "up": np.array([0, 1], dtype=np.int64),
    "right": np.array([1, 0], dtype=np.int64),
    "down": np.array([0, -1], dtype=np.int64),
    "left": np.array([-1, 0], dtype=np.int64),
}

DIR_REVERSE = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}

# Discrete action table: (rotation_degrees, movement_cells)
# rotation in {-90, 0, +90}, movement clipped to [-3, 3]
DEFAULT_ACTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (-90, 0),
    (90, 0),
    (0, 1),
    (0, 2),
    (0, 3),
)


@dataclass
class MazeSpec:
    maze_path: Optional[str] = None
    random_dim: int = 12
    random_extra_openings: int = 0
    random_seed: Optional[int] = None


class MazeInstance:
    def __init__(self, spec: MazeSpec):
        self.spec = spec
        self._tmp_path: Optional[str] = None
        self._maze = self._build_maze(spec)
        self._steps = 0
        self.visitation = np.zeros((self._maze.dim, self._maze.dim), dtype=int)

    def infos(self):
        result = dict()
        unique = (self.visitation > 0).astype(float)
        result['coverage'] = unique.mean()
        result['effective_coverage'] = unique.sum()/min(self._steps + 1, self.dim * self.dim)
        return result

    def reset(self, location=None):
        self.visitation.fill(0)
        if location is not None:
            self.visitation[*location] = 1
        self._steps = 0

    def add_step(self):
        self._steps += 1

    @property
    def step(self):
        return self._steps

    @property
    def maze(self) -> Maze:
        return self._maze

    @property
    def dim(self) -> int:
        return self._maze.dim

    @property
    def goal_bounds(self) -> Tuple[int, int]:
        # goal is center 2x2 block
        return (self._maze.dim // 2 - 1, self._maze.dim // 2)

    def close(self) -> None:
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.remove(self._tmp_path)
            except OSError:
                pass
        self._tmp_path = None

    def _build_maze(self, spec: MazeSpec) -> Maze:
        if spec.maze_path:
            return Maze(spec.maze_path)

        dim = int(spec.random_dim)
        if dim % 2 != 0:
            raise ValueError("random_dim must be even")
        if dim <= 1:
            raise ValueError("random_dim must be > 1")

        walls = generate_maze(
            dim,
            seed=spec.random_seed,
            extra_openings=int(spec.random_extra_openings),
        )
        fd, path = tempfile.mkstemp(prefix=f"maze_{dim}_", suffix=".txt")
        os.close(fd)
        write_maze(path, dim, walls)
        self._tmp_path = path
        return Maze(path)


class MazeRenderer:
    def __init__(self, maze: Maze, max_steps: int, action_table: Sequence[Tuple[int, int]]):
        import pygame
        from image import Image

        self._pygame = pygame
        self._maze = maze
        self._done = False
        self._game_objects = []
        self._thread: Optional[threading.Thread] = None
        self._status_cell = [self._maze.dim - 2, -1]

        pygame.init()
        size = 50 * maze.dim + 100
        screen = pygame.display.set_mode((size, size))
        self.image = Image(maze.dim, screen)
        self._game_objects.append(self.image)
        self._draw_static_walls()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _draw_static_walls(self) -> None:
        # Draw only right/up walls once to avoid duplicates.
        dim = self._maze.dim
        for x in range(dim):
            for y in range(dim):
                cell = [x, y]
                if x + 1 < dim and not self._maze.is_permissible(cell, "right"):
                    self.image.draw_line_between(np.array([x, y]), np.array([x + 1, y]))
                if y + 1 < dim and not self._maze.is_permissible(cell, "up"):
                    self.image.draw_line_between(np.array([x, y]), np.array([x, y + 1]))

    def set_status(self, *, step: int, action_idx: int, rotation: int, movement: int) -> None:
        del action_idx
        text = f"t={int(step)} a=({int(rotation)},{int(movement)})"
        # Force runtime font render on each update (temporary debug mode).
        self.image.text_cache.pop(text, None)
        self.image.update_text(self._status_cell, text)

    def _loop(self) -> None:
        clock = self._pygame.time.Clock()
        while not self._done:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self._done = True
            dt_ms = clock.tick(30)
            for obj in self._game_objects:
                obj.update(dt_ms)
            self._pygame.display.update()

    def reset_pose(self, location: np.ndarray, heading_idx: int) -> None:
        self.image.reset_arrow()
        self.image.move(location.astype(np.float64))
        self.set_status(step=0, action_idx=-1, rotation=0, movement=0)
        if heading_idx == 1:
            self.image.rotate(-np.pi / 2.0)
        elif heading_idx == 2:
            self.image.rotate(np.pi)
        elif heading_idx == 3:
            self.image.rotate(np.pi / 2.0)

    def update_pose(self, location: np.ndarray, prev_heading_idx: int, heading_idx: int) -> None:
        self.image.move(location.astype(np.float64))
        delta = (int(heading_idx) - int(prev_heading_idx)) % 4
        if delta == 1:
            self.image.rotate(-np.pi / 2.0)
        elif delta == 3:
            self.image.rotate(np.pi / 2.0)
        elif delta == 2:
            self.image.rotate(np.pi)

    def close(self) -> None:
        if self._thread is None:
            return
        self._done = True
        time.sleep(0.05)
        self._thread.join(timeout=1.0)
        self._pygame.quit()
        self._thread = None


class MazeVecEnv:
    """Vectorized maze environment.

    Observation dict (all batched with leading dimension `num_envs`):
    - `sensor`: float32, shape [N, 3], left/front/right wall distances.
    - `heading_idx`: int64, shape [N], heading in {0:up, 1:right, 2:down, 3:left}.
    - `location`: int64, shape [N, 2], x/y location.
    - `step_count`: int64, shape [N], steps since last reset.

    `reward` is always zero here (intrinsic rewards should be added by trainer).

    Step API semantics (roughly aligned with ManagerBasedRLEnv):
    1. Process actions (rotation/movement).
    2. Advance environment dynamics.
    3. Render (if enabled).
    4. Update counters and compute rewards/terminations.
    5. Reset environments that terminated/truncated (when `auto_reset=True`).
    6. Compute observations.
    7. Return observations, rewards, done flags, and extras (`info`).

    Note: with `auto_reset=True`, returned observations for done envs are post-reset
    observations, while rewards correspond to the transition that just completed.
    """

    def __init__(
        self,
        num_envs: int = 1,
        maze_path: Optional[str] = None,
        random_dim: int = 12,
        random_extra_openings: int = 0,
        randomize_each_reset: bool = False,
        max_steps: int = 1000,
        action_table: Sequence[Tuple[int, int]] = DEFAULT_ACTIONS,
        render: bool = False,
        seed: Optional[int] = None,
        auto_reset: bool = True,
        return_torch: bool = True,
        device: str = "cpu",
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if render and num_envs != 1:
            raise ValueError("render=True currently supports only num_envs=1")

        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.action_table = list(action_table)
        self.randomize_each_reset = bool(randomize_each_reset)
        self.auto_reset = bool(auto_reset)
        self.return_torch = bool(return_torch)
        self.device = torch.device(device)

        self._base_spec = MazeSpec(
            maze_path=maze_path,
            random_dim=random_dim,
            random_extra_openings=random_extra_openings,
            random_seed=seed,
        )
        self._rng = np.random.default_rng(seed)
        self._mazes: List[MazeInstance] = []
        self._renderer: Optional[MazeRenderer] = None

        self._location = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._heading_idx = np.zeros((self.num_envs,), dtype=np.int64)
        self._step_count = np.zeros((self.num_envs,), dtype=np.int64)

        self._build_mazes()
        if render:
            self._renderer = MazeRenderer(
                self._mazes[0].maze,
                max_steps=self.max_steps,
                action_table=self.action_table,
            )

    @property
    def action_space_n(self) -> int:
        return len(self.action_table)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        for maze_inst in self._mazes:
            maze_inst.close()
        self._mazes = []

    def _make_spec_for_env(self, env_idx: int) -> MazeSpec:
        del env_idx
        if self._base_spec.maze_path:
            return MazeSpec(maze_path=self._base_spec.maze_path)

        return MazeSpec(
            maze_path=None,
            random_dim=self._base_spec.random_dim,
            random_extra_openings=self._base_spec.random_extra_openings,
            random_seed=int(self._rng.integers(0, 2**31 - 1)),
        )

    def _build_mazes(self) -> None:
        for maze_inst in self._mazes:
            maze_inst.close()
        self._mazes = [MazeInstance(self._make_spec_for_env(i)) for i in range(self.num_envs)]

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.randomize_each_reset:
            self._build_mazes()

        self._location.fill(0)
        self._heading_idx.fill(0)
        self._step_count.fill(0)
        for i, maze in enumerate(self._mazes):
            maze.reset(self._location[i])

        if self._renderer is not None:
            self._renderer.reset_pose(self._location[0], int(self._heading_idx[0]))

        return self._build_obs()

    def _heading_name(self, env_idx: int) -> str:
        return HEADINGS[int(self._heading_idx[env_idx])]

    def _goal_hit(self, env_idx: int) -> bool:
        x, y = self._location[env_idx]
        a, b = self._mazes[env_idx].goal_bounds
        return bool((x in (a, b)) and (y in (a, b)))

    def _sensors(self, env_idx: int) -> np.ndarray:
        maze = self._mazes[env_idx].maze
        heading = self._heading_name(env_idx)
        loc = self._location[env_idx].tolist()
        vals = [maze.dist_to_wall(loc, d) for d in DIR_SENSORS[heading]]
        return np.asarray(vals, dtype=np.float32)

    def _apply_rotation(self, env_idx: int, rotation: int) -> None:
        if rotation == -90:
            self._heading_idx[env_idx] = (self._heading_idx[env_idx] - 1) % 4
        elif rotation == 90:
            self._heading_idx[env_idx] = (self._heading_idx[env_idx] + 1) % 4

    def _apply_movement(self, env_idx: int, movement: int) -> Tuple[int, bool]:
        maze = self._mazes[env_idx].maze
        movement = int(np.clip(movement, -3, 3))
        moved = 0
        blocked = False

        while movement != 0:
            heading = self._heading_name(env_idx)
            if movement > 0:
                if maze.is_permissible(self._location[env_idx].tolist(), heading):
                    self._location[env_idx] += DIR_MOVE[heading]
                    movement -= 1
                    moved += 1
                else:
                    blocked = True
                    break
            else:
                rev = DIR_REVERSE[heading]
                if maze.is_permissible(self._location[env_idx].tolist(), rev):
                    self._location[env_idx] += DIR_MOVE[rev]
                    movement += 1
                    moved -= 1
                else:
                    blocked = True
                    break

        if not blocked:
            self._mazes[env_idx].visitation[*self._location[env_idx]] += 1
        return moved, blocked

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(-1)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}")

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        infos: List[Dict] = []

        for i in range(self.num_envs):
            self._mazes[i].add_step() 
            action_idx = int(actions[i])
            if action_idx < 0 or action_idx >= len(self.action_table):
                raise ValueError(f"Invalid action index {action_idx} for env {i}")

            prev_heading = int(self._heading_idx[i])
            rotation, movement = self.action_table[action_idx]

            self._apply_rotation(i, int(rotation))
            moved, blocked = self._apply_movement(i, int(movement))
            self._step_count[i] += 1

            hit_goal = self._goal_hit(i)
            timeout = bool(self._step_count[i] >= self.max_steps)
            terminated[i] = False
            truncated[i] = timeout

            if self._renderer is not None and i == 0:
                self._renderer.set_status(
                    step=int(self._step_count[i]),
                    action_idx=action_idx,
                    rotation=int(rotation),
                    movement=int(movement),
                )
                self._renderer.update_pose(self._location[i], prev_heading, int(self._heading_idx[i]))

            info = {
                "hit_goal": bool(hit_goal),
                "timeout": bool(timeout),
                "blocked": bool(blocked),
                "moved": int(moved),
                "rotation": int(rotation),
                "movement": int(movement),
                "location": self._location[i].copy(),
                "heading_idx": int(self._heading_idx[i]),
                "sensor": self._sensors(i),
                "step_count": int(self._step_count[i]),
            }
            info.update(self._mazes[i].infos())
            infos.append(info)

            if self.auto_reset and (terminated[i] or truncated[i]):
                self._single_reset(i)

        done = terminated | truncated
        info = {
            "terminated": self._to_output(terminated, dtype=torch.bool),
            "truncated": self._to_output(truncated, dtype=torch.bool),
            "episode_length_buf": self._to_output(self._step_count.copy(), dtype=torch.float32),
            "per_env": infos,
        }
        obs = self._build_obs()
        return obs, self._to_output(rewards, dtype=torch.float32), self._to_output(done, dtype=torch.bool), info

    def _single_reset(self, env_idx: int) -> None:
        if self.randomize_each_reset:
            self._mazes[env_idx].close()
            self._mazes[env_idx] = MazeInstance(self._make_spec_for_env(env_idx))

        self._location[env_idx] = 0
        self._heading_idx[env_idx] = 0
        self._step_count[env_idx] = 0
        self._mazes[env_idx].reset(self._location[env_idx])

        if self._renderer is not None and env_idx == 0:
            self._renderer.reset_pose(self._location[0], int(self._heading_idx[0]))

    def _to_output(self, arr: np.ndarray, dtype: torch.dtype):
        if self.return_torch:
            return torch.as_tensor(arr, dtype=dtype, device=self.device)
        return arr

    def _build_obs(self):
        sensor = np.stack([self._sensors(i) for i in range(self.num_envs)], axis=0)
        obs = {
            "sensor": sensor,
            "heading_idx": self._heading_idx.copy(),
            "location": self._location.copy(),
            "step_count": self._step_count.copy(),
        }
        if not self.return_torch:
            return obs
        return {
            "sensor": torch.as_tensor(obs["sensor"], dtype=torch.float32, device=self.device),
            "heading_idx": torch.as_tensor(obs["heading_idx"], dtype=torch.long, device=self.device),
            "location": torch.as_tensor(obs["location"], dtype=torch.long, device=self.device),
            "step_count": torch.as_tensor(obs["step_count"], dtype=torch.long, device=self.device),
        }


class RandomAgent:
    """Simple pluggable agent for smoke testing."""

    def __init__(self, action_space_n: int, seed: Optional[int] = None):
        self.action_space_n = int(action_space_n)
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        batch = int(obs["sensor"].shape[0])
        return self._rng.integers(0, self.action_space_n, size=(batch,), dtype=np.int64)
