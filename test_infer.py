"""Self-check for frame sampling and the MPS isin workaround: uv run python test_infer.py"""
import os
import tempfile

import cv2
import numpy as np
import torch

from infer import sample_uniform
from run_eval import score

assert score("The wallet, then the can.", "can|wallet") == "wallet"  # first mention wins
assert score("B)", "a|b") == "b"
assert score("Two.", "1|2|3") == "2"
assert score("I am not sure.", "yes|no") == ""

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "count.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for i in range(120):
        w.write(np.full((64, 64, 3), i * 2, np.uint8))  # brightness encodes the frame index
    w.release()
    video, meta = sample_uniform(path, 8)
    assert meta.frames_indices == [7, 22, 37, 52, 67, 82, 97, 112], meta.frames_indices
    assert video.shape == (8, 64, 64, 3)
    # the decoded frames are the ones we asked for, in order
    assert np.allclose(video.mean(axis=(1, 2, 3)), np.array(meta.frames_indices) * 2, atol=3)

assert torch.isin(torch.tensor([1, 5, 9], device="mps"), torch.tensor([5, 9], device="mps")).tolist() == [False, True, True]
print("ok")
