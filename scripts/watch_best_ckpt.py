"""
Watch training_log.txt and maintain checkpoint-best (lowest train loss)
alongside the rolling checkpoint-last that the Trainer keeps.

Usage (run in background while training):
    python scripts/watch_best_ckpt.py --out_dir <work_dir> &

The Trainer is configured with save_total_limit=1, so it only ever keeps
one numbered checkpoint (the latest). This watcher:
  1. Parses {"loss": X, ...} lines from training_log.txt
  2. When a new numbered checkpoint appears and its step loss < best so far,
     copies it to checkpoint-best (replacing the previous best).
  3. Exits when training_log.txt contains "training finished" or the
     numbered checkpoint stops changing for --idle_secs seconds.
"""

import argparse
import json
import os
import re
import shutil
import time


def latest_numbered_ckpt(out_dir):
    ckpts = [
        d for d in os.listdir(out_dir)
        if re.match(r"checkpoint-\d+$", d)
        and os.path.isdir(os.path.join(out_dir, d))
    ]
    if not ckpts:
        return None, -1
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    name = ckpts[-1]
    return os.path.join(out_dir, name), int(name.split("-")[1])


def parse_losses(log_path):
    """Return dict {step: loss} from trainer log lines."""
    losses = {}
    if not os.path.exists(log_path):
        return losses
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = re.search(r"\{'loss':\s*([\d.]+).*?'step':\s*(\d+)", line)
            if m:
                losses[int(m.group(2))] = float(m.group(1))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--idle_secs", type=int, default=600,
                    help="Exit after this many seconds with no new checkpoint")
    ap.add_argument("--poll_secs", type=int, default=30)
    args = ap.parse_args()

    log_path = os.path.join(args.out_dir, "training_log.txt")
    best_dir = os.path.join(args.out_dir, "checkpoint-best")
    best_loss = float("inf")
    best_step = -1
    last_ckpt_step = -1
    idle_since = time.time()

    print(f"[WATCH] Monitoring {args.out_dir}", flush=True)
    print(f"[WATCH] Will update checkpoint-best whenever train loss improves", flush=True)

    while True:
        time.sleep(args.poll_secs)

        # Check for training finished
        if os.path.exists(log_path):
            with open(log_path, errors="ignore") as f:
                content = f.read()
            if "Full training finished" in content or "training finished" in content.lower():
                print("[WATCH] Training finished marker detected.", flush=True)
                break

        losses = parse_losses(log_path)
        ckpt_path, ckpt_step = latest_numbered_ckpt(args.out_dir)

        if ckpt_step > last_ckpt_step and ckpt_path is not None:
            idle_since = time.time()
            last_ckpt_step = ckpt_step

            # Find the loss at or just before this checkpoint step
            relevant = {s: l for s, l in losses.items() if s <= ckpt_step}
            if relevant:
                step_loss = relevant[max(relevant)]
                print(f"[WATCH] checkpoint-{ckpt_step}  loss={step_loss:.4f}  "
                      f"best_so_far={best_loss:.4f}", flush=True)
                if step_loss < best_loss:
                    best_loss = step_loss
                    best_step = ckpt_step
                    tmp = best_dir + ".tmp"
                    if os.path.exists(tmp):
                        shutil.rmtree(tmp)
                    shutil.copytree(ckpt_path, tmp)
                    if os.path.exists(best_dir):
                        shutil.rmtree(best_dir)
                    os.rename(tmp, best_dir)
                    print(f"[WATCH] ✓ checkpoint-best updated  "
                          f"step={best_step}  loss={best_loss:.4f}", flush=True)
                else:
                    print(f"[WATCH] checkpoint-{ckpt_step} not better than "
                          f"best (step={best_step}, loss={best_loss:.4f}), skip.", flush=True)
        else:
            if time.time() - idle_since > args.idle_secs:
                print(f"[WATCH] No new checkpoint for {args.idle_secs}s — exiting.", flush=True)
                break

    print(f"[WATCH] Done. Best checkpoint: step={best_step}  loss={best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
