"""
main.py
────────
CyberPulse AI – unified CLI entry point.

Usage examples
──────────────
  # Full pipeline (fetch → preprocess → train)
  python main.py --all

  # Step by step
  python main.py --fetch
  python main.py --preprocess
  python main.py --train

  # Just train LSTM, skip Prophet (faster)
  python main.py --train --no-prophet

  # Launch dashboard
  python main.py --dashboard
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CyberPulse AI",
        description="Proactive Cyber Threat Forecasting System",
    )
    parser.add_argument("--all",        action="store_true", help="Run full pipeline: fetch → preprocess → train")
    parser.add_argument("--fetch",      action="store_true", help="Fetch raw CVE data from NIST NVD API")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess raw CVE data")
    parser.add_argument("--train",      action="store_true", help="Train LSTM + Prophet models")
    parser.add_argument("--dashboard",  action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--no-lstm",    action="store_true", help="Skip LSTM training")
    parser.add_argument("--no-prophet", action="store_true", help="Skip Prophet training")
    return parser.parse_args()


def step_fetch():
    log.info("═══ STEP 1: Fetching CVE data from NIST NVD API ═══")
    from pipeline.data_fetcher import fetch_all
    fetch_all()


def step_preprocess():
    log.info("═══ STEP 2: Preprocessing CVE data ═══")
    from pipeline.preprocessor import run
    run()


def step_train(use_lstm: bool = True, use_prophet: bool = True):
    log.info("═══ STEP 3: Training models (LSTM=%s, Prophet=%s) ═══", use_lstm, use_prophet)
    from pipeline.trainer import train_all
    metrics = train_all(use_lstm=use_lstm, use_prophet=use_prophet)
    log.info("Training complete. Metrics summary:")
    for atk, m in metrics.items():
        if m:
            parts = "  |  ".join(f"{k}={v:.2f}" for k, v in m.items())
            log.info("  %-15s  %s", atk, parts)


def step_dashboard():
    log.info("═══ Launching Streamlit dashboard ═══")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"],
        check=False,
    )


def main():
    args = parse_args()

    if not any(vars(args).values()):
        # Default: show help
        print(
            "\nCyberPulse AI – no arguments provided.\n"
            "Quick start:\n"
            "  python main.py --all          # full pipeline\n"
            "  python main.py --dashboard    # launch dashboard\n"
            "\nRun with --help for all options.\n"
        )
        return

    if args.all or args.fetch:
        step_fetch()

    if args.all or args.preprocess:
        step_preprocess()

    if args.all or args.train:
        step_train(
            use_lstm=not args.no_lstm,
            use_prophet=not args.no_prophet,
        )

    if args.dashboard:
        step_dashboard()


if __name__ == "__main__":
    main()
