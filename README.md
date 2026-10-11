# Preflight

**Know whether a robot will succeed before it moves.** Entry for the [Nebius Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) — **Physical AI track**.

> Build embodied and edge agents that sense and act in the real world: robotics, IoT, and on-device
> intelligence, powered by Nemotron, GR00T, Cosmos and Sonic models and coordinated through an agent
> runtime. Use Nebius Serverless Jobs to run simulations, generate synthetic data, evaluate robot
> policies, or process sensor data at scale, and Nebius Serverless Endpoints to serve real-time
> inference when needed.

## Track requirements checklist

- [ ] NVIDIA model(s): Nemotron / GR00T / Cosmos / Sonic
- [ ] Agent runtime coordinating the models
- [ ] Nebius **Serverless Jobs** — sim / synthetic data / policy eval / sensor processing at scale
- [ ] Nebius **Serverless Endpoints** — real-time inference
- [ ] Demo video with ≥1 min of hardware operating (or key application modules in action)

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in Nebius / Tavily keys
uv run npai info
uv run pytest
```

Hackathon rules digest: [`docs/hackathon-rules.md`](docs/hackathon-rules.md). Design spec: [`docs/specs/2026-09-30-preflight-design.md`](docs/specs/2026-09-30-preflight-design.md).

## Closed-loop learning architecture

The next architecture step is a closed-loop SO-101 learner: a world-model layer predicts candidate outcomes, FLUX 3 Action or GR00T executes short action chunks behind a safety controller, and failures feed an asynchronous DAgger/recovery loop. The current evidence and open work are summarised in [`docs/closed-loop-architecture.md`](docs/closed-loop-architecture.md), with the Mermaid source in [`docs/specs/2026-10-09-closed-loop-world-model-architecture.mmd`](docs/specs/2026-10-09-closed-loop-world-model-architecture.mmd).
