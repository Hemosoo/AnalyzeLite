"""Answer a question about a video with SmolVLM2-500M, using N uniformly sampled frames.

    uv run infer.py clip.mov "Is the person waving? Answer yes or no." --frames 8
"""
import argparse
import resource
import time

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE, DTYPE = "mps", torch.float32  # float16 output turns to garbage at 16 frames; macOS 13 MPS has no bfloat16

# MPS before macOS 14 rejects torch.isin on integer tensors, which generate() uses. Drop this on macOS 14+.
_isin = torch.isin
torch.isin = lambda e, t, **kw: (
    (e.unsqueeze(-1) == torch.as_tensor(t, device=e.device).reshape(-1)).any(-1)
    if e.device.type == "mps" and not e.is_floating_point()
    else _isin(e, t, **kw)
)


def sample_uniform(path, n):
    """Decode the frame at the centre of each of n equal segments of the clip."""
    cap = cv2.VideoCapture(path)
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        raise SystemExit(f"Can't read video: {path}")
    want = ((np.arange(n) + 0.5) * total / n).astype(int).tolist()
    frames, got = [], []
    for i in range(want[-1] + 1):
        if not cap.grab():  # the header's frame count can overshoot
            break
        if i in want:
            frames.append(cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2RGB))
            got.append(i)
    cap.release()
    meta = VideoMetadata(total_num_frames=total, fps=fps, duration=total / fps, frames_indices=got)
    return np.stack(frames), meta


def load():
    processor = AutoProcessor.from_pretrained(MODEL)
    return AutoModelForImageTextToText.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE), processor


def ask(model, processor, path, question, n):
    """Return the answer, the sampled frames' metadata, sampling seconds and end-to-end seconds."""
    t0 = time.perf_counter()
    video, meta = sample_uniform(path, n)
    t1 = time.perf_counter()
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=[[video]], video_metadata=[meta], return_tensors="pt").to(DEVICE, dtype=DTYPE)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
    return answer, meta, t1 - t0, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("question")
    p.add_argument("--frames", type=int, default=8)
    args = p.parse_args()

    model, processor = load()
    answer, meta, t_sample, t_total = ask(model, processor, args.video, args.question, args.frames)

    print(answer)
    print(f"{len(meta.timestamps)} frames at {' '.join(f'{s:.1f}s' for s in meta.timestamps)}")
    # shortcut: MPS has no peak counter; its allocator keeps freed blocks cached, so this reads near peak.
    # Poll it from a thread during generate if the numbers look off.
    mps_gb = torch.mps.driver_allocated_memory() / 2**30
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(f"end-to-end {t_total:.2f}s (sampling {t_sample:.2f}s) | MPS mem {mps_gb:.2f} GB | peak RSS {rss_gb:.2f} GB")


if __name__ == "__main__":
    main()
