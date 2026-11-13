"""Answer a question about a video with SmolVLM2-500M, using N uniformly sampled frames.

    uv run infer.py clip.mov "Is the person waving? Answer yes or no." --frames 8
    uv run infer.py              # interactive: the model stays loaded, ask one question after another
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
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
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    encode = model.model.get_image_features

    def encode_frame_by_frame(pixel_values, pixel_attention_mask=None, **kwargs):
        # One frame at a time through the vision encoder, so its working memory doesn't grow with the frame count.
        # Batch size is always 1 here: pixel_values is (1, frames, 3, H, W).
        frames = pixel_values.split(1, dim=1)
        masks = [None] * len(frames) if pixel_attention_mask is None else pixel_attention_mask.split(1, dim=1)
        features = [encode(f, m, **kwargs).pooler_output[0] for f, m in zip(frames, masks)]
        return BaseModelOutputWithPooling(pooler_output=[torch.cat(features)])

    model.model.get_image_features = encode_frame_by_frame
    return model, processor


def ask(model, processor, path, question, n):
    """Return the answer, the sampled frames' metadata, sampling seconds, end-to-end seconds and peak GPU GB."""
    t0 = time.perf_counter()
    video, meta = sample_uniform(path, n)
    t1 = time.perf_counter()
    peak, done = [torch.mps.current_allocated_memory()], threading.Event()

    def watch():  # shortcut: MPS has no peak counter, so poll it; a spike shorter than 5 ms could be missed
        while not done.wait(0.005):
            peak[0] = max(peak[0], torch.mps.current_allocated_memory())

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, videos=[[video]], video_metadata=[meta], return_tensors="pt").to(DEVICE, dtype=DTYPE)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        t2 = time.perf_counter()
    finally:  # stop the watcher even if generation fails, or the process can't exit
        done.set()
        watcher.join()
    torch.mps.empty_cache()  # hand cached blocks back, so between questions only the weights stay in memory
    return answer, meta, t1 - t0, t2 - t0, peak[0] / 2**30


def report(answer, meta, t_sample, t_total, peak_gb):
    print(answer)
    print(f"{len(meta.timestamps)} frames at {' '.join(f'{s:.1f}s' for s in meta.timestamps)}")
    print(f"end-to-end {t_total:.2f}s (sampling {t_sample:.2f}s) | GPU peak {peak_gb:.2f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?")
    p.add_argument("question", nargs="?", help="leave out to keep the model loaded and ask questions interactively")
    p.add_argument("--frames", type=int, default=8)
    args = p.parse_args()

    model, processor = load()
    if args.question:
        return report(*ask(model, processor, args.video, args.question, args.frames))

    video = args.video
    while True:
        try:
            # Enter keeps the last video; drag-and-drop pastes paths quoted or with escaped spaces
            video = input(f"video{f' [{video}]' if video else ''}: ").strip().strip("'\"").replace("\\ ", " ") or video
            question = input("question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if video and question:
            try:
                report(*ask(model, processor, os.path.expanduser(video), question, args.frames))
            except SystemExit as e:  # unreadable video: say so and keep the model loaded
                print(e)


if __name__ == "__main__":
    main()
