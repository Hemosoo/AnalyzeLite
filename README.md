# AnalyzeLite

On-device video question answering with SmolVLM2-500M: how many frames does a compact VLM need to keep its accuracy?

## Setup

Apple Silicon Mac. Install [uv](https://docs.astral.sh/uv/), then everything else installs on the first `uv run`. The first run also downloads the model (about 1 GB).

## Run

```
uv run infer.py path/to/clip.mov "Was the drawer opened and then closed? Answer yes or no." --frames 8
```

It prints the answer, the timestamps of the sampled frames, and end-to-end latency (including frame sampling) plus memory.

## Evaluate

```
uv run run_eval.py ~/Downloads experiments/results.csv --frames 4 8 16
```

This asks every question in `questions.csv` about its clip, scores the answers against the ground truth, and logs latency and memory for each. Results so far are in `experiments/`.

## Test

```
uv run python test_infer.py
```

This checks that the sampler returns the right frames in time order, and that the macOS 13 GPU workaround works. It prints `ok`.

## macOS 13 notes

torch is pinned to 2.8, the last version that can use the Apple GPU on macOS 13. The model runs in float32 because float16 output breaks down at 16 frames. After upgrading to macOS 14 or later, the pin and the `torch.isin` workaround in `infer.py` can go.
