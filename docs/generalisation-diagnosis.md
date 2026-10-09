# Generalisation diagnosis: SO-101 policy on new scenes, objects and tasks

Goal: raise v5's generalisation on the internal benchmark (six paired suites, 96 rehearsals each) and on the external
[so101-benchmark](https://github.com/Fatsharkovo/so101-benchmark). This doc tracks each experiment, its PR and its
verdict. Per-PR detail is in the PR bodies.

## Where v5 stands

| Suite | v5 |
|---|---|
| held_out_objects | 52–55 / 96 |
| seen_vocabulary | 75 |
| clean_many | 78–81 |
| clean_top_down_many | 82–83 |
| real_1_pomade_tin_many | 73–74 |
| in_path_pomade_tin_many | 73–82 |
| so101-benchmark `stack_blue_on_red` | **0 / 50** |

Ranges are two runs of the same checkpoint. Re-runs move a cell by up to ±9, so only paired McNemar verdicts count.

## Experiments

| # | Hypothesis | PR | Verdict |
|---|---|---|---|
| 1 | Failures are skill, not visuals: label them | [#12](https://github.com/charleneleong-ai/nebius-physical-ai/pull/12) | 576 rehearsals: no_contact 41%, pushed 25%, dropped 17%. Two thirds are reaching and gripping. |
| 2 | Targeted demos fix those modes (v6: recoveries + lookalike decoys) | [#13](https://github.com/charleneleong-ai/nebius-physical-ai/pull/13) | **Within noise.** no_contact −13 and held-out +8, but pushes +16: it reaches the right object, then mis-grasps. |
| 3 | The external 0/50 is a visual gap | [#14](https://github.com/charleneleong-ai/nebius-physical-ai/pull/14) | **Confirmed.** Joint mapping, front camera and arm colour match; the table texture and wrist mount differ. v5 never heads for the cubes. |
| 4 | DAgger on the policy's own failure states | [#15](https://github.com/charleneleong-ai/nebius-physical-ai/pull/15) | **Round 1: KEEP, within noise** (−2 pooled). no_contact 57 → 45 and knocked_place 11 → 0, but dropped 24 → 33 and holding 4 → 13. Its 85 recoveries were only 3.5% of training. |
| 5 | Upweight the recoveries (round 2: 157 recoveries ×8 against 2,400 base, about 34% of training) | [#15](https://github.com/charleneleong-ai/nebius-physical-ai/pull/15) | **PROMOTE: pooled +26, no suite worse.** clean_many 81 → 86, real_1_pomade_tin_many 74 → 86 (p=0.004), in_path 73 → 81. no_contact 57 → 33, knocked_place 11 → 0–1. held_out_objects unchanged (52 → 50), and dropped/holding stay high. |
| 6 | Visual DR closes the external gap (v7: procedural floors, wrist mount randomised) | [#17](https://github.com/charleneleong-ai/nebius-physical-ai/pull/17) | **Mostly the wrist mount.** The Seeed mount alone drops v5 from 81 to 15 on clean_many's scenes; v7 lifts that to 53 (+42/−4). But v7 regresses internally (clean_many 81 → 58) and reaches only 1/50 externally. |
| 7 | A camera-robust world action model transfers (FLUX 3 Action, zero-shot) | [#18](https://github.com/charleneleong-ai/nebius-physical-ai/pull/18) | **Zero-shot: no.** 0/16 on clean_many against v5's 10/16 on the same scenes, 0/10 on so101-benchmark. LoRA v8 (3,600 demos incl. visual DR): clean_many 1/96, held_out 10/96 against v5's 81 and 52; 0/10 externally. A fair run on v5's own 800 demos at v5's volume is training. Its imagined futures barely improve on zero-shot (L1 15.4 vs 16.5/255): FLUX already sees our scene, it maps it to the wrong actions. |

## What the evidence says so far

- **Targeted on-policy data is the lever, if it is weighted.** DAgger round 2 is the first change to beat v5 (+26 pooled), with recoveries at a third of training. Round 1's identical loop at 3.5% was flat, as were v5b and v6.
- **Data volume is not the lever.** v5b (more steps on the same data) and v6 (2× the data, targeted) both land within
  noise of v5. This matches PlugCharger, where 30× the demos moved imitation from 0.07 to 0.17 only.
- **The bottleneck has moved from target choice to grasping.** v6 cut no_contact but raised pushed. The scripted expert
  always grasps top-down at the centre, so demos never show a corrected grasp. Recovery demos teach re-approaching but
  not a better first contact.
- **The external benchmark is mostly a wrist-camera problem.** The appearance suites isolate it: on clean_many's own
  scenes, the Seeed wrist mount costs v5 66 of 81 successes, the textured floor 12. Randomising the mount in training
  recovers 38 of those, so the axis is learnable; v7's remaining external failures point at the front-camera framing.
- **Continuing a converged checkpoint on shifted inputs costs skill.** v7 (v5 + visual DR at lr 2e-5) lost 23 on
  clean_many. Wrist views from two mounts are one of only two inputs; mixing them late overwrote the one being scored.

## Next moves

1. **Round 3 of DAgger from the promoted round-2 checkpoint** (#15). Upweighting is what made it work: the same loop at 3.5% recoveries was flat. Drops need a recovery the expert can record from a lifted state.
2. **The fair FLUX run** (#18): LoRA on v5's 800 demos at v5's volume, same paired suites, plus an open-loop audit that separates a model failure from a bridge failure. Results around 2026-10-10 06:00 UTC.
3. **Train with the wrist mount randomised from the start, from the hqfang base**, instead of continuing v5. That should keep v7's +38 on the Seeed mount without its internal regression.
4. **The rest of the external gap**: with the wrist mount handled internally, v7 still scores 1/50. Compare so101-benchmark's front-camera framing, start pose and episode length next.
5. **Grasp quality is now the main internal failure** (pushed and dropped). The scripted expert always grasps top-down at the centre; varied grasp offsets in its demos would show the policy corrected grasps.
