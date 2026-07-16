#!/usr/bin/env python
"""Start the voice. Ctrl-C to stop her.

    ./yap                # forever
    ./yap --seconds 90   # a taste
    ./yap --quiet        # don't print what she's saying
"""

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from yapper.config import ROOT, load
from yapper.pipeline import Pipeline
from yapper.player import Player
from yapper.voice import Voice
from yapper.writer import Writer

LOG = ROOT / "yapper.log"


class Announcing:
    """Prints each story at the moment she starts speaking it.

    Not when she writes it -- she runs several stories ahead, so printing at
    write time made the text race the voice by minutes.
    """

    def __init__(self, player):
        self.player = player

    def play(self, chunk):
        print(f"\n{chunk.text}\n", flush=True)
        self.player.play(chunk)


def server_up(url):
    try:
        return requests.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


def start_server(cfg):
    """Bring up llama-server if it isn't already running.

    It stays a separate process on purpose: restarting the yapper then costs
    nothing instead of reloading 10GB.
    """
    if server_up(cfg.llm.server_url):
        print("llama-server: already up")
        return None
    if not cfg.llm.server_binary.exists():
        sys.exit(f"No llama-server at {cfg.llm.server_binary} (LLAMA_SERVER in .env)\n"
                 f"Build it:  scripts/build-llama.sh")

    print("llama-server: starting (this takes ~30s the first time)...")
    port = cfg.llm.server_url.rsplit(":", 1)[-1]
    proc = subprocess.Popen(
        [str(cfg.llm.server_binary), "-m", cfg.llm.model_path,
         "-ngl", str(cfg.llm.n_gpu_layers), "-c", str(cfg.llm.ctx_size),
         "--host", "127.0.0.1", "--port", port],
        stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        if server_up(cfg.llm.server_url):
            print("llama-server: ready")
            return proc
        if proc.poll() is not None:
            sys.exit(f"llama-server died on startup. See {LOG}")
        time.sleep(1)
    sys.exit(f"llama-server didn't come up. See {LOG}")


def main():
    ap = argparse.ArgumentParser(description="A voice that doesn't stop.")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--quiet", action="store_true", help="don't print the text")
    args = ap.parse_args()

    cfg = load()
    if not Path(cfg.voice.reference).exists():
        sys.exit("She doesn't exist yet. Run:\n"
                 "    ./design-voice generate\n"
                 "    ./design-voice play 1\n"
                 "    ./design-voice choose 1")

    proc = start_server(cfg)
    stop_event = threading.Event()

    writer = Writer(cfg)
    voice = Voice(cfg)
    speakers = Player(cfg, stop_event=stop_event)
    player = speakers if args.quiet else Announcing(speakers)

    pipe = Pipeline(
        writer, voice, player,
        text_depth=cfg.pipeline.text_queue_depth,
        audio_depth=cfg.pipeline.audio_queue_depth,
        retry_base=cfg.pipeline.retry_base,
        retry_max=cfg.pipeline.retry_max,
        log=lambda m: print(m, file=sys.stderr, flush=True),
    )

    def shutdown(*_):
        print("\nstopping...", flush=True)
        stop_event.set()
        pipe.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\nShe's warming up. First words in ~20 seconds.\n")
    pipe.start()

    try:
        if args.seconds:
            time.sleep(args.seconds)
            shutdown()
        else:
            while any(t.is_alive() for t in pipe.threads):
                time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
    finally:
        stop_event.set()
        pipe.stop()
        speakers.close()
        # llama-server is left running on purpose: the next ./yap starts instantly.
        if proc is not None:
            print("(llama-server left running; kill it with: pkill llama-server)")


if __name__ == "__main__":
    main()
