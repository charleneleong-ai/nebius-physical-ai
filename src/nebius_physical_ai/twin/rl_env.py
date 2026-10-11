"""A state-based pick-and-place environment on the twin, to train an RL expert (scripts/rl_expert.py has why).

The policy sees privileged simulator state, not cameras, so training is fast and needs no GPU: arm joints and
velocities, the gripper frame, the pick and place blocks, their sizes and the vectors between them. It moves the arm by
small joint deltas at 10 Hz. The reward is staged like ManiSkill's PickCube: reach the pick block, lift it, carry it
over the place block, stack it (the twin's own `StackCheck`, which ends the episode).

Episodes start either from a fresh scene or from a DAgger failure snapshot: the policy's own last moment in a failed
rollout (pushed, dropped, held), which is where the scripted expert recovers only 40% of the time.

A grasp, both jaws touching the pick block, counts as progress: without it PPO settled on hovering near the block for
the reach reward and never closed the gripper. Progress is paid as potential-based shaping, so holding the block above
the target forever (which out-earned a one-off stack under plain dense rewards) earns nothing. With `residual`, the scripted expert's plan from the episode's start
state is the base action and the policy only adds a small bounded correction (residual RL, as in Policy Decorator,
arXiv 2412.13630), so a zero policy is the scripted expert and learning only has to find the recoveries it misses.
"""

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from nebius_physical_ai.twin.dagger import FailureSnapshot
from nebius_physical_ai.twin.expert import PickPlaceExpert, SceneSampler, UnreachableTarget
from nebius_physical_ai.twin.rollout import CONTROL_HZ, LIFTED_M, StackCheck
from nebius_physical_ai.twin.scene import ARM_JOINTS, BlockPhysics, Twin, TwinLayout

POLICY_HZ = 10
MAX_DELTA_RAD = 0.06  # per policy step, about 34 degrees a second at full stick
SUCCESS_BONUS = 10.0
GAMMA = 0.99  # PPO's discount, so the shaping leaves the optimal policy unchanged
RESIDUAL_RAD = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.4])  # largest correction per joint (gripper moves most)
RESIDUAL_COST = 0.05
JAWS = ("gripper", "moving_jaw_so101_v1")
OBS_SIZE = 6 + 6 + 3 * 8 + 2 + 6 + 1


class PickPlaceEnv(gym.Env):
    """Place the pick block on the place block, from state. Fresh scenes re-size and move the blocks of one compiled
    template (`Twin.variant`); each snapshot's scene is compiled once and put back in its saved state on reset."""

    def __init__(
        self,
        seed: int = 0,
        horizon_s: float = 15.0,
        hard: bool = False,
        snapshots: list[FailureSnapshot] | None = None,
        snapshot_share: float = 0.5,
        residual: bool = False,
    ) -> None:
        self.layout = TwinLayout()
        self.sampler = SceneSampler.hard(self.layout, seed) if hard else SceneSampler(self.layout, seed)
        self.template = Twin.from_objects(self.sampler.sample(), self.layout, start_pose=self.sampler.start_pose())
        self.snapshots, self.snapshot_share = snapshots or [], snapshot_share if snapshots else 0.0
        self.restored: dict[int, Twin] = {}
        self.rng = np.random.default_rng(seed)
        self.horizon, self.residual = round(horizon_s * POLICY_HZ), residual
        self.twin, self.check = self.template, StackCheck()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(ARM_JOINTS),), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """`options={"snapshot": i}` starts from snapshot i; otherwise a snapshot with probability `snapshot_share`."""
        super().reset(seed=seed)
        index = (options or {}).get("snapshot")
        if index is None and self.rng.random() < self.snapshot_share:
            index = int(self.rng.integers(len(self.snapshots)))
        if index is None:
            objects, start = self.sampler.sample(), self.sampler.start_pose()
            self.twin, self.check = self.template.variant(objects, self.layout, BlockPhysics(), start), StackCheck()
        else:
            self.twin, self.check = self.from_snapshot(index), StackCheck(*self.pick_place(index))
        self.site, self.limits = self.twin.model.site("gripperframe").id, self.twin.arm_limits()
        self.target, self.steps, self.tick = self.twin.arm_qpos(), 0, 0
        self.lifted = self.height() > LIFTED_M
        self.plan = self.expert_plan() if self.residual else [self.target]
        self.last_potential = self.potential()
        return self.observe(), {"snapshot": index}

    def expert_plan(self) -> list[np.ndarray]:
        """The scripted expert's joint targets, one per control tick, from where the episode starts (the arm holds its
        pose if the expert can't reach a waypoint)."""
        expert = PickPlaceExpert(self.twin, "", self.check.top, self.check.bottom)
        try:
            segments = expert.waypoints()
        except UnreachableTarget:
            return [self.twin.arm_qpos()]
        plan, start = [], self.twin.arm_qpos()
        for arm, grip, seconds in segments:
            goal = np.append(arm, grip)
            plan += list(np.linspace(start, goal, max(1, round(seconds * CONTROL_HZ)) + 1)[1:])
            start = goal
        return plan

    def base(self) -> np.ndarray:
        return self.plan[min(self.tick, len(self.plan) - 1)]

    def grasped(self) -> bool:
        """Both jaws in contact with the pick block."""
        model, data = self.twin.model, self.twin.data
        pick, touching = model.body(self.check.top).id, set()
        for c in data.contact[: data.ncon]:
            bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if pick in bodies:
                touching |= {model.body(b).name for b in bodies}
        return all(jaw in touching for jaw in JAWS)

    def pick_place(self, index: int) -> tuple[str, str]:
        s = self.snapshots[index]
        return s.pick, s.place

    def from_snapshot(self, index: int) -> Twin:
        s = self.snapshots[index]
        if index not in self.restored:
            self.restored[index] = s.restore(self.layout)
        twin = self.restored[index]
        twin.data.qpos[:], twin.data.qvel[:] = s.qpos, s.qvel
        twin.data.ctrl[: len(ARM_JOINTS)] = twin.arm_qpos()
        mujoco.mj_forward(twin.model, twin.data)
        return twin

    def gripper(self) -> np.ndarray:
        return self.twin.data.site_xpos[self.site].copy()

    def height(self) -> float:
        """How far the pick block is above resting on the table (the table top is under the place block)."""
        pick, place = self.twin.block_pos(self.check.top), self.twin.block_pos(self.check.bottom)
        table = place[2] - self.twin.geometry(self.check.bottom).half[2]
        return float(pick[2] - table - self.twin.geometry(self.check.top).half[2])

    def stack_point(self) -> np.ndarray:
        """Where the pick block's centre sits when stacked."""
        place = self.twin.block_pos(self.check.bottom)
        rise = self.twin.geometry(self.check.bottom).half[2] + self.twin.geometry(self.check.top).half[2]
        return place + np.array([0.0, 0.0, rise])

    def observe(self) -> np.ndarray:
        grip, pick, place = self.gripper(), self.twin.block_pos(self.check.top), self.twin.block_pos(self.check.bottom)
        qvel = np.array([self.twin.data.joint(j).qvel[0] for j in ARM_JOINTS])
        sizes = [np.asarray(self.twin.geometry(n).half, dtype=float) for n in (self.check.top, self.check.bottom)]
        vectors = [pick - grip, self.stack_point() - pick, place - grip]
        phase = min(self.tick / max(len(self.plan), 1), 1.0)
        flags = [self.lifted, self.grasped()]
        state = [self.twin.arm_qpos(), qvel, grip, pick, place, *sizes, *vectors, flags, self.base(), [phase]]
        return np.concatenate(state).astype(np.float32)

    def potential(self) -> float:
        """Progress through the task: reach, then grasp, then lift and carry towards the stack point."""
        grip, pick = self.gripper(), self.twin.block_pos(self.check.top)
        self.lifted |= self.height() > LIFTED_M
        reach = 1 - np.tanh(10 * np.linalg.norm(pick - grip))
        if not self.lifted:
            return float(reach + self.grasped())
        return float(2 + reach + 2 * (1 - np.tanh(10 * np.linalg.norm(self.stack_point() - pick))))

    def reward(self) -> tuple[float, bool]:
        """Potential-based shaping (Ng et al. 1999) plus the success bonus: paying the change in progress rather than
        progress itself, so hovering at the stack point earns nothing and only stacking pays off."""
        previous, now = self.last_potential, self.potential()
        self.last_potential = now
        if self.check(self.twin):
            return SUCCESS_BONUS, True
        return float(GAMMA * now - previous), False

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1, 1)
        lo, hi = self.limits.T
        if not self.residual:
            self.target = np.clip(self.target + MAX_DELTA_RAD * action, lo, hi)
        for _ in range(CONTROL_HZ // POLICY_HZ):
            if self.residual:  # the expert's target this tick, corrected
                self.target = np.clip(self.base() + RESIDUAL_RAD * action, lo, hi)
            self.twin.step_towards(self.target, 1 / CONTROL_HZ)
            self.tick += 1
        self.steps += 1
        reward, success = self.reward()
        if self.residual:
            reward -= RESIDUAL_COST * float(np.square(action).mean())
        truncated = self.steps >= self.horizon
        return self.observe(), reward, success, truncated, {"is_success": success, "lifted": self.lifted}

    def close(self) -> None:
        for twin in [self.template, *self.restored.values()]:
            twin.close()
