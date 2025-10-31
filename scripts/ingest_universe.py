"""CLI wrapper to run the backend ingestion from the repo root.

Usage:
    python scripts/ingest_universe.py --limit 50
"""
import argparse
from backend.ingest import ingest_universe


def main():
    parser = argparse.ArgumentParser("ingest_universe")
    parser.add_argument("--limit", type=int, default=50, help="Number of tickers to ingest")
    args = parser.parse_args()
    res = ingest_universe(limit=args.limit)
    print(res)


if __name__ == "__main__":
    main()
