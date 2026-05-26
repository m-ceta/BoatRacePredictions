from __future__ import annotations

import argparse
import time
import webbrowser


DEFAULT_REPEAT = 12
DEFAULT_INTERVAL_SECONDS = 3600


def run(url: str, repeat: int = DEFAULT_REPEAT, interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> None:
    target_url = url.strip()
    if not target_url:
        raise SystemExit("No URL was provided.")
    for _ in range(repeat):
        webbrowser.open(target_url)
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    args = parser.parse_args()
    run(url=args.url, repeat=args.repeat, interval_seconds=args.interval_seconds)
