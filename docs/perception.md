# Cosmos 3 and marker-corrected perception

`npai preflight` takes a phone frame to a verdict. The ArUco markers place every object on the mat in millimetres;
Cosmos adds labels, plausibility notes and a cross-check. The model is **Cosmos 3 Nano**
(`nvidia/Cosmos3-Nano`, released 31 May 2026; it needs no Hugging Face token), hosted on the A100 next to GR00T.

## How perception works

1. `ScenePerceiver.perceive` finds the corner markers, fits the frame-to-mat homography and segments the objects on
   the rectified mat. Positions, footprints, yaw and colour come from here, never from the vision model.
2. The vision model (Cosmos, or `--tf-vl` for Token Factory) returns labels, `issues` and a `bbox_2d` per object,
   normalised to 0-1000. Each box centre is projected through the same homography and compared with the markers' view.
3. A task block (blue, red) the vision model does not confirm drops to confidence 0.5; anything it sees that the
   markers' view lacks becomes a warning. Distractors stay in the twin, so clutter shows up in the rehearsal.
4. Perception fidelity is `markers / 4 x least task-block confidence x (1 - marker fit / 5 mm)`. Below 0.6 the answer
   is ASK_HUMAN and no rehearsal runs. The same applies when fewer than two markers are visible or a task block is
   missing.

Cosmos 3 does not always answer in the shape the prompt asks for. On a real photo it returned a bare fenced list of
detections with no wrapper object; on other prompts a reasoning model can put chain of thought, with stray braces and
brackets, ahead of the JSON. `parse_scene_json` takes the last top-level answer, either an object with `objects` or a
list of detections that each carry a `bbox_2d`; anything else (a list of labels, prose) is an error, not a guess.

### Verified: `bbox_2d` is 0-1000, `(x1, y1, x2, y2)`

`scripts/verify_bbox.py` reads Cosmos 3 Nano's boxes under four candidate conventions and compares each box centre
with the object's true pixel position (from the markers and the colour segmentation, which do not depend on the vision
model). Run on 2026-10-05 against five frames with markers, the 0-1000 `(x1, y1, x2, y2)` reading won every time:

- Synthetic frames (2 and 4 boxes): median 1.3 px, about 0.4 mm on the mat.
- Real phone-style photos, 3 boxes each (pomade tin, glasses case, ball cluster): median 8 to 10 px, about 3 to 4 mm.
  Part of that is parallax: a box centre on a tall object sits above the mat plane the homography assumes.
- The other readings are far off: 0-1 is not meaningful, pixels of the image sent is 34 to 68 mm out, and swapped
  axes 16 to 34 mm out.

The cross-check tolerance is 50 mm, so the real-photo error leaves a wide margin. To re-check after changing the model
or the prompt: `uv run python scripts/verify_bbox.py frame.jpg ...` (`--tf-vl` for Token Factory); it exits 1 when 0-1000
is not the best fit.

Limits: object heights are unknown from one frame (the twin assumes as tall as narrow, 20-60 mm); the 5 mm marker
tolerance is a heuristic tuned on synthetic frames (clean fit 0.3 mm, a marker 18 mm off its corner 2 mm); Cosmos
lists fewer objects than the markers' segmentation on real photos (3 of 5 or 6, it skips small or pale ones), so
the `bbox_2d` cross-check mostly confirms the blue and red task blocks rather than every distractor.

```
uv run npai preflight frame.jpg --port 5557 --n 32                  # Cosmos if COSMOS_ENDPOINT_URL is set, else Token Factory
uv run npai preflight frame.jpg --tf-vl --offline-explain --n 16    # force Token Factory (runs on credits)
uv run npai preflight frame.jpg --cosmos --n 16                     # force Cosmos
uv run npai preflight frame.jpg --markers-only --n 16               # no vision model at all
uv run npai preflight frame.jpg --camera configs/front_camera.json  # twin camera from `npai calibrate-camera`
uv run npai preflight frame.jpg --pick "pomade tin" --place "glasses case"   # other objects, named as Cosmos names them
```

### Choosing the objects

The default task is the one the GR00T checkpoints were fine-tuned on: pick the blue cube, place it on the red cube,
with the prompt `Pick blue cube and place on red cube`. `--pick` and `--place` name any two objects instead, and
`--prompt` sets the instruction the policy is sent (by default it is derived from the names).

A name is matched by whole words. First against the colour label the markers give each blob (`blue`, `yellow`,
`dark`; the largest wins when several match), then against the words Cosmos uses for the object in its own box: `ball`
finds `multicolored ball cluster`, and the blob under that box becomes the task block, named `ball` in the twin. A name
that matches nothing stops the run with ASK_HUMAN and lists what the markers and Cosmos saw. A plain colour the
markers do not see is rejected before any vision call.

The policy has only ever been trained on the blue and red cubes, so any other task is out of distribution. The run
says so in its notes and in the W&B config (`in_distribution`), and the result then measures how that policy copes
with the new objects, not how a policy trained for them would.

## Hosting Cosmos 3 on the A100

The A100 (Ampere) is on NVIDIA's supported list for Cosmos 3 (Ampere, Hopper, Blackwell). `scripts/cosmos_local.sh`
runs `vllm/vllm-openai:cosmos3` with `nvidia/Cosmos3-Nano`. The model is not gated, so no Hugging Face token is needed.

```
mise run cosmos-up       # start the container; the first start downloads the weights
mise run cosmos-health   # container state and the served model
mise run cosmos-logs     # last log lines
mise run cosmos-down     # remove the container (weights stay cached)
```

- **GPU memory:** capped at 30 percent of the card (`COSMOS_GPU_FRACTION`, default 0.30, about 24 GB), so the GR00T
  servers can share the GPU. The Nano reasoner needs roughly 16 to 17 GB.
- **Disk:** about 54 GB for the first start (19 GB image plus 35 GB of weights in `~/.cache/cosmos3`). Check `df -h`
  first; fine-tune checkpoints are about 22 GB each.
- **Network:** published on `127.0.0.1:8001` only. There is no authentication, so do not expose the port.
- **Settings:** `COSMOS_PORT`, `COSMOS_MODEL_ID`, `COSMOS_IMAGE`, `COSMOS_MAX_MODEL_LEN`, `COSMOS_CACHE` and
  `COSMOS_NAME` override the defaults. `scripts/cosmos_local.sh print` shows the exact `docker run` command.

On the A100, put this in `.env` (the client appends `/v1`):

```
COSMOS_ENDPOINT_URL=http://127.0.0.1:8001
COSMOS_MODEL=cosmos
```

## Not in this PR: a Nebius endpoint

Cosmos could also run on a Nebius Serverless endpoint, which the hackathon prefers to the A100. That was prototyped
(`nebius ai endpoint create` with the same image and arguments as `scripts/cosmos_local.sh`) and removed from this PR:
the AI Cloud tenant is suspended, there is no GPU credit, and NVIDIA's Cosmos 3 GPU list names Ampere, Hopper and
Blackwell, not the L40S that is the cheapest Nebius GPU, so an H100 (about $4.50/h) would be the safe choice. The
Token Factory `Cosmos3-Super-Reasoner` listed in the console returned 404 for our key on 2026-10-05, so the
`--tf-vl` fallback uses Qwen3.8 instead.
