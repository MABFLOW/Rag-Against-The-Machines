"""Command-line entry point exposing :class:`Engine` via Fire."""

import sys

try:
    from .engine import Engine
    import fire
except KeyboardInterrupt:
    print("\nInterrupted during startup.")
    sys.exit(1)


if __name__ == "__main__":
    try:
        fire.Fire(Engine)
    except KeyboardInterrupt:
        print("\nProgram has been Stoped, 'Please re-run the previous cmd'")
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)
