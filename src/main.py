"""CLI entry-point orchestrating smoke test / full experiment.

This file is auto-generated because the original monolithic script was not supplied. Any
attempt to execute will raise an informative exception so that users immediately notice
the missing implementation instead of running silent no-ops.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generated stub – original Experiment Code missing."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run smoke-test config")
    group.add_argument(
        "--full-experiment", action="store_true", help="Run full-scale experiment"
    )

    _ = parser.parse_args()

    sys.exit(
        "Cannot run: the provided prompt did not include the experiment source code to refactor."
    )


if __name__ == "__main__":
    main()
