# Hackathon rules digest

Source: [rules](https://nebiusglobalaihackathon.devpost.com/rules) · [updates](https://nebiusglobalaihackathon.devpost.com/updates) (fetched 2026-09-30).

## Deadline

**Fri 30 Oct 2026, 10:00 PT (17:00 UTC).** Judging from Nov–Dec; winners ~11 Jan 2027.

## Hard requirements (Stage One pass/fail)

- Runs on Nebius — runtime call to Token Factory **or** deployed on AI Cloud (Serverless Jobs / Endpoints / DevPods).
- Uses ≥1 NVIDIA open model (Nemotron, GR00T, Cosmos, Sonic).
- Genuine attempt at the track goal, not a rebrand.
- **Public** repo with Apache-2.0/MIT/MPL-2.0 license shown in About; README covers setup + how NVIDIA models / Token Factory / Nebius tools are used.
- Video < 3 min, public YouTube, no copyrighted music; Physical AI needs ≥1 min of hardware operating (or key modules in action). No demo URL needed for this track.
- Text description + **mandatory feedback** section on Nebius/NVIDIA tools.
- Existing projects allowed only if the submission-period changes are explained — keep this repo new.

## Stage Two — four equally weighted criteria (1–5)

Tie-break in this order: **Technological Implementation** → **Design** (complete product, not a PoC) →
**Potential Impact** (specific real audience) → **Quality of Idea** (non-obvious).

Organiser hint: *"scripted demo is table stakes; hardware reacting live to real-world input is where this track can really shine."*

## Prizes

Grand $20k / 2nd $10k / 3rd $6k; track winner = Jetson Orin Nano. One Overall **or** one Track award, **plus** one bonus:
Best Use of Tavily ($3k, runtime Tavily call), Most Valuable Feedback ($100), City Winner (London attendees only).

## Credits & models

- $25 Token Factory: code `NEBIUS-DEVPOST-GLOBAL26`; +$25 TF + $25 Tavily via [Nebius Builders](https://dev.nebius.com/builders).
- No GPU credit advertised — AI Cloud is pay-as-you-go.
- Token Factory: Nemotron 3 Ultra / Super / Nano, Nemotron 3.5 Lightning; VLMs (Qwen2.5-VL-72B etc).
  Serverless Cosmos reasoner was **retired Aug 2026** — Cosmos Reason 2 now runs as a container on AI Cloud GPUs.
- Reference tooling: [nebius/nebius-physical-ai](https://github.com/nebius/nebius-physical-ai) (robot SDG → LeRobot, MuJoCo physics judge, SkyPilot).
