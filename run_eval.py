"""Ask every question in questions.csv about its clip, score the answers and log latency and memory.

    uv run run_eval.py ~/Downloads experiments/results.csv --frames 4 8 16
"""
import argparse
import csv
import os
import re
from statistics import mean

from infer import ask, load

NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}


def score(output, choices):
    """Return the first of the |-separated choices the output mentions, or "" if it mentions none."""
    options = choices.split("|")
    for word in re.findall(r"[a-z0-9]+", output.lower()):
        word = NUMBERS.get(word, word)
        if word in options:
            return word
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips_dir")
    p.add_argument("out_csv")
    p.add_argument("--frames", type=int, nargs="+", default=[8])
    args = p.parse_args()

    with open("questions.csv") as f:
        questions = list(csv.DictReader(f))
    model, processor = load()
    clip = lambda q: os.path.join(args.clips_dir, q["clip"])
    ask(model, processor, clip(questions[0]), questions[0]["question"], 8)  # warm up GPU kernels, not recorded

    rows = []
    with open(args.out_csv, "w", newline="") as f:
        for n in args.frames:
            for q in questions:
                output, meta, t_sample, t_total, peak_gb = ask(model, processor, clip(q), q["question"], n)
                pred = score(output, q["choices"])
                rows.append(row := {
                    **q, "budget": n, "frames": len(meta.timestamps), "output": output, "pred": pred,
                    "correct": int(pred == q["answer"]), "sampling_s": round(t_sample, 3), "latency_s": round(t_total, 3),
                    "gpu_peak_gb": round(peak_gb, 2),
                })
                if len(rows) == 1:
                    w = csv.DictWriter(f, fieldnames=row.keys())
                    w.writeheader()
                w.writerow(row)
                f.flush()  # keep partial results if the run dies
                print(f"[{n}f] {'✓' if row['correct'] else '✗'} {q['clip']} {q['question']} -> {output!r}", flush=True)

    for n in args.frames:
        rs = [r for r in rows if r["budget"] == n]
        first_choice = mean(r["choices"].split("|")[0] == r["answer"] for r in rs)
        print(f"\n{n} frames: accuracy {mean(r['correct'] for r in rs):.0%} (always-first-choice {first_choice:.0%}) | "
              f"latency {mean(r['latency_s'] for r in rs):.2f}s | GPU peak {max(r['gpu_peak_gb'] for r in rs):.2f} GB")
        for t in sorted({r["type"] for r in rs}):
            tr = [r for r in rs if r["type"] == t]
            print(f"  {t:13s} {mean(r['correct'] for r in tr):4.0%}  ({len(tr)} questions)")


if __name__ == "__main__":
    main()
