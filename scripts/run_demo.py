#!/usr/bin/env python3
"""SIH-26168 Interactive Navigation Prototype Runner.

Launches the local HTTP dashboard server and connects the web frontend
to the authoritative Step 17 Python EndToEndNavigationPipeline.

Usage:
    python scripts/run_demo.py [--port 8080] [--scenario mixed_urban] [--auto-open]
"""

import os
import sys
import argparse
import webbrowser
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.demo.server import DemoServer


def parse_args():
    parser = argparse.ArgumentParser(description="SIH-26168 Interactive Prototype Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--scenario", type=str, default="mixed_urban", help="Initial trajectory scenario (default: mixed_urban)")
    parser.add_argument("--outage-duration", type=float, default=30.0, help="Initial outage duration in seconds (default: 30.0)")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open browser on launch")
    return parser.parse_args()


def main():
    args = parse_args()

    # Pre-flight environment check
    ckpt_path = os.path.join("results", "ml_training", "best_drift_model.pt")
    bench_path = os.path.join("results", "final_benchmark", "benchmark_summary.json")

    print("================================================================================")
    print("       SIH-26168 AI-ML INTELLIGENT DEAD RECKONING SYSTEM PROTOTYPE")
    print("================================================================================")
    print(f"[*] Verifying Navigation Core Artifacts:")
    print(f"    - TCN Model Weights:      {'FOUND (' + ckpt_path + ')' if os.path.exists(ckpt_path) else 'NOT FOUND'}")
    print(f"    - Benchmark Database:     {'FOUND (' + bench_path + ')' if os.path.exists(bench_path) else 'NOT FOUND'}")
    print(f"    - System Classification:  YELLOW (Software Validated | Physical Vehicle Trials Pending)")
    print(f"    - Provenance:             SYNTHETIC_DATA / OFFLINE_LOG_REPLAY")

    # Initialize demo server
    server = DemoServer(
        host=args.host,
        port=args.port,
        checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None,
    )

    server.session.load_scenario(args.scenario, args.outage_duration)
    server.start(auto_replay_loop=True)

    url = server.get_url()
    print(f"\n[+] Local Dashboard Active: {url}")
    print(f"[+] Controls Available in UI:")
    print(f"    - [START] / [PAUSE] / [RESET] / [STEP]")
    print(f"    - [SIMULATE GNSS OUTAGE] / [RESTORE GNSS]")
    print(f"    - [RUN FULL SIH DEMO] (Automated 60s Guided Presentation)")
    print(f"    - Real-Time Feature Toggles: AI TCN, NHC, ZUPT, Road Map Matching")
    print(f"\n[*] Press Ctrl+C to terminate the server.\n")

    if args.auto_open:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[*] Shutting down Demo Server...")
        server.shutdown()
        print("[+] Server stopped cleanly.")


if __name__ == "__main__":
    main()
