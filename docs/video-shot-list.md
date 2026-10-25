# Demo video — shot list

The hackathon's hard requirement: **under 3 minutes, public YouTube, no copyrighted music, and at least one
minute of hardware operating or key application modules in action.** The FAQ confirms simulation counts for the
Physical AI track, so the three Spaces are the footage — no live A100 session is needed, and the Nebius key stays
where it belongs.

Target 2:45, which leaves margin under the limit.

## Before recording

- Open all three Spaces and let them warm. The first hit on [`chaleong/preflight`](https://huggingface.co/spaces/chaleong/preflight)
  is slow because it is a docker Space.
- Have two mat photos ready: one that clears the gate, one that does not.
- Screen-record at 1080p or better.
- Shots 2–5 are roughly 95 seconds of modules in action, comfortably over the one-minute floor.

Scenes worth filming, chosen for what each one shows rather than for a particular score — the rehearsal
numbers move with every gallery render, so read the current ones off the Space before recording:

| Scene | Typical verdict | Why it is worth a shot |
|---|---|---|
| `centre` | GO | the clean pass |
| `tight_gap` | RETRY | marginal, so the gate is visibly doing work |
| `too_close` | RETRY | a hard refusal, almost every rehearsal fails |
| `cluttered` | ASK_HUMAN | perception refuses before any rehearsal happens |
| `swapped_task` | — | same blocks, instruction reversed: the one scene where only the *task* is out of distribution |

`hidden_marker` reads as a GO despite its name, because three of the four markers still fit the mat. It is not
the ASK_HUMAN shot — use `cluttered`.

## Shots

| # | Time | On screen | Voice-over |
|---|---|---|---|
| 1 | 0:00–0:14 | Title card, then a phone photo of the A3 mat | A robot policy will attempt almost anything you ask it. Preflight checks whether it *should* — before the arm moves. |
| 2 | 0:14–0:38 | Live: drag the photo onto the demo Space; markers lock on, the scene panel fills | One photo. ArUco markers place a MuJoCo twin on the mat, and Cosmos 3, served through Nebius Token Factory, reads what is actually in the scene. |
| 3 | 0:38–1:12 | Gallery: play two or three rollout videos and let them loop | Then GR00T rehearses the task in that twin — dozens of times, with the objects jittered. This is the policy trying, and sometimes failing, before anything real happens. |
| 4 | 1:12–1:38 | A GO card, then an ASK_HUMAN card; hold on the explanation | A trust gate scores the rehearsals. Here it clears. Here it refuses — and Nemotron says why, in a sentence an operator can act on. |
| 5 | 1:38–2:08 | Autopilot Space: the loop diagram, then the cycle cards | Every refusal is a training signal. Nemotron proposes the next experiment from a menu of recipes; a paired McNemar gate decides whether it ships. Nothing is promoted on a hunch. |
| 6 | 2:08–2:32 | The results table and the funnel diagram | Eight hundred rollouts harvested, 197 recoverable failures, 59 recovered by a scripted expert — retrained to 483 of 576 against a 438 baseline. New objects: 57 of 96, up from 52. |
| 7 | 2:32–2:45 | Stack card and links | GR00T, Cosmos 3 and Nemotron — NVIDIA open models on Nebius. The external benchmark is still near zero, and that is in the report. Everything is public. |

On-screen text, so the models are legible with the sound off: `Cosmos 3` at 0:20, `GR00T N1.7` at 0:45,
`Nemotron` at 1:25, `Nebius Token Factory` at 2:35.

## Two deliberate choices

**Shot 7 states the external benchmark is still 0–1/50.** It could be left out inside three minutes and nobody
would notice, but [`docs/report.pdf`](report.pdf) and [`docs/experiment-card.md`](experiment-card.md) both report
it, and the entry that volunteers its own negative result is the one that reads as honest. It costs four seconds.

**Shot 6 quotes the funnel, not only the win.** "483 of 576" on its own is marketing; `800 → 197 → 59 → 483` is
the result. The README carries every denominator, so the video should match it.
