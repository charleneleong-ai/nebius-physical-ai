# nebius-physical-ai

Entry for the [Nebius Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) — **Physical AI track**.

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

Design specs live in [`docs/specs/`](docs/specs/).
