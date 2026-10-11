# Closed-loop World-model Architecture Implementation Plan

> **For agentic workers:** This documentation-only plan records the architecture contract for future implementation work.

**Goal:** Document how Preflight's existing simulation, policy, failure-harvesting, and benchmark components compose into a safe closed-loop SO-101 learner.

**Architecture:** A perception layer feeds a world-model layer composed of Cosmos visual prediction and Isaac/MuJoCo grounded simulation. FLUX 3 Action or GR00T proposes short action chunks behind a safety controller; outcomes enter an asynchronous DAgger/recovery and promotion loop.

**Tech Stack:** Mermaid, Markdown, MuJoCo, Isaac Sim, Cosmos 3 Nano, GR00T, FLUX 3 Action, LeRobot, DAgger, GitHub PRs.

**Spec:** `docs/specs/2026-10-09-closed-loop-world-model-architecture.mmd` and `docs/closed-loop-architecture.md`

## Global Constraints

- Cosmos-generated visual futures are not treated as physically valid action labels without validation.
- Weight updates happen asynchronously and are promoted only through fixed internal and external benchmarks.
- The physical action path retains joint, velocity, force, workspace, collision, and emergency-stop limits.
- The architecture PR is documentation-only; it does not claim real-arm or online-learning completion.

## Review Focus

- World-model latency must not block the fast control loop; the diagram separates predictive and action timescales.
- Cosmos visual plausibility must not be confused with physical correctness; Isaac/MuJoCo remains the grounded validation path.
- Online data collection must not silently promote a regressed checkpoint; benchmark evaluation and rollback remain explicit.
- The progress table must distinguish measured scores from pending experiments and must link each claim to the relevant PR.
- FLUX and GR00T must remain replaceable policy backends rather than being coupled to the world-model interface.

### Task 1: Architecture diagram and component boundaries

**Files:**
- Create: `docs/specs/2026-10-09-closed-loop-world-model-architecture.mmd`
- Create: `docs/closed-loop-architecture.md`

- [x] Add the perception → world model → policy → safety → robot feedback path.
- [x] Add the asynchronous episode buffer → recovery → training → evaluation → promotion path.
- [x] Document Cosmos, Isaac/MuJoCo, FLUX, GR00T, DAgger, and human intervention responsibilities.

### Task 2: Progress evidence and repository entry point

**Files:**
- Modify: `README.md`
- Create: `docs/plans/2026-10-09-closed-loop-architecture.md`

- [x] Link the architecture and progress summary from the README.
- [x] Summarise PRs #11–#18 with measured internal and external results.
- [x] State the real-arm and learned-world-model milestones that remain open.

### Task 3: Documentation verification and handoff

- [ ] Run `git diff --check`.
- [ ] Run `pre-commit run --all-files`.
- [ ] Review the rendered PR body for links, Mermaid source, tables, and explicit pending items.
- [ ] Open the documentation-only PR against `main`.
