# Hackathon rules digest

Source: [rules](https://nebiusglobalaihackathon.devpost.com/rules) · [updates](https://nebiusglobalaihackathon.devpost.com/updates) ·
[FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs) (rules fetched 2026-09-30, FAQ 2026-10-07).

## Deadline

**Fri 30 Oct 2026, 10:00 PT (17:00 UTC).** Judging from Nov–Dec; winners ~11 Jan 2027.

## Hard requirements (Stage One pass/fail)

- Runs on Nebius — runtime call to Token Factory **or** deployed on AI Cloud (Serverless Jobs / Endpoints / DevPods).
  The FAQ confirms that calling an NVIDIA model through Token Factory satisfies this, and that the frontend and other
  backend parts may run elsewhere.
- Uses ≥1 NVIDIA open model (Nemotron, GR00T, Cosmos, Sonic; the FAQ adds others such as Alpamayo).
- An NVIDIA model reached through another provider (Hugging Face inference, OpenRouter) does **not** count (FAQ).
- Genuine attempt at the track goal, not a rebrand.
- **Public** repo with Apache-2.0/MIT/MPL-2.0 license shown in About; README covers setup + how NVIDIA models / Token Factory / Nebius tools are used.
- Video < 3 min, public YouTube, no copyrighted music; Physical AI needs ≥1 min of hardware operating (or key modules in action). No demo URL needed for this track.
  The FAQ confirms simulation is acceptable for Physical AI (no hardware list required).
- Text description + **mandatory feedback** section on Nebius/NVIDIA tools.
- Existing projects allowed only if the submission-period changes are explained — keep this repo new.

## Submission checklist

Freeze model work around **Oct 20**; the last ten days are for the demo, video and write-up.

**Compliance**
- [x] NVIDIA open models: GR00T (policy), Cosmos 3 (perception), Nemotron (verdict explanations)
- [x] Token Factory at runtime: [`NemotronAgent`](../src/nebius_physical_ai/pipeline.py) explains every verdict; a
  missing key raises instead of falling back to canned text
- [x] No NVIDIA model through a third-party provider: Hugging Face is used only to download weights we run ourselves
- [x] Simulation counts for Physical AI (FAQ); the SO-101 twin carries the demo until the arm arrives
- [x] New repo (started 2026-09-30), Apache-2.0 in [`LICENSE`](../LICENSE)
- [ ] Every live verdict in the demo goes through Token Factory (no `--offline-explain` in the recorded run)
- [ ] Rotate the Nebius API key that was shared in chat, and scan the git history for secrets
- [ ] Repo public, licence shown in the GitHub About panel
- [ ] Re-read NVIDIA's model licences (GR00T N1.7 base: research and evaluation use) against the submission

**README**
- [x] Setup and quickstart
- [x] How Nebius and NVIDIA are used (README "How Preflight uses Nebius and NVIDIA")
- [x] Model licences
- [ ] Results summary with links to the W&B runs and the benchmark
- [ ] Demo instructions for the live loop (phone page → verdict → explanation)

**Demo and submission**
- [ ] Live loop end to end on the A100 with v5 served and Nemotron answering: phone frame → twin → rehearsal → verdict
- [ ] Video < 3 min on public YouTube, ≥1 min of the system working (the arm if it arrives, else the twin), no
  copyrighted music; name Token Factory, Nemotron, GR00T and Cosmos on screen or in voice-over
- [ ] Devpost text description
- [ ] Mandatory feedback section on Nebius and NVIDIA tools (notes so far: Token Factory credits, AI Cloud billing
  availability, GR00T N1.6 vs N1.7 tooling, LeRobot GR00T checkpoint units, Cosmos 3 serving)
- [ ] Submit before Oct 30, 10:00 PT (aim for Oct 29)

**Optional**
- [ ] One Nebius AI Cloud piece (e.g. a Serverless Job running a batch of rehearsals) to show Nebius beyond the API
- [ ] Tavily runtime call (Best Use of Tavily bonus)

## Stage Two — four equally weighted criteria (1–5)

Tie-break in this order: **Technological Implementation** → **Design** (complete product, not a PoC) →
**Potential Impact** (specific real audience) → **Quality of Idea** (non-obvious).

Organiser hint: *"scripted demo is table stakes; hardware reacting live to real-world input is where this track can really shine."*

## Prizes

Grand $20k / 2nd $10k / 3rd $6k; track winner = Jetson Orin Nano. One Overall **or** one Track award, **plus** one bonus:
Best Use of Tavily ($3k, runtime Tavily call), Most Valuable Feedback ($100), City Winner (London attendees only, in person).

## Credits & models

- $25 Token Factory: code `NEBIUS-DEVPOST-GLOBAL26`; +$25 TF + $25 Tavily via [Nebius Builders](https://dev.nebius.com/builders).
  The FAQ confirms these are Token Factory credits only; there is no AI Cloud credit grant.
- Token Factory: Nemotron 3 Ultra / Super / Nano, Nemotron 3.5 Lightning; VLMs (Qwen2.5-VL-72B etc).
  Serverless Cosmos reasoner was **retired Aug 2026** — Cosmos Reason 2 now runs as a container on AI Cloud GPUs.
- Reference tooling: [nebius/nebius-physical-ai](https://github.com/nebius/nebius-physical-ai) (robot SDG → LeRobot, MuJoCo physics judge, SkyPilot).
