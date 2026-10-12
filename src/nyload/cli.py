"""nyload: single entry point for the project.

    nyload doctor                      check Python / torch / GPU / data
    nyload data  --years 2013-2023     download + build data (see `nyload data -h`)
    nyload train [overrides...]        train, e.g. `nyload train model=lstm train.epochs=50`
    nyload eval  [overrides...]        backtest, e.g. `nyload eval run_dir=outputs/runs/<run>`
    nyload runs                        list training runs and their test metrics
    nyload test                        run the test suite

`train` and `eval` pick the env config for this machine (local_win / local_wsl / runpod / auto)
unless `env=...` is given.
"""
import json
import os
import platform
import sys
from pathlib import Path

from nyload import paths

COMMANDS = ("doctor", "data", "train", "eval", "runs", "test")


def detect_env() -> str:
    import torch

    if not torch.cuda.is_available():
        return "auto"
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    if platform.system() == "Windows":
        return "local_win"
    if "microsoft" in platform.release().lower():
        return "local_wsl"
    return "auto"


def _hydra(entry, args: list[str], prog: str) -> None:
    if not any(a.startswith("env=") for a in args) and not any(a in ("-h", "--help", "--cfg") for a in args):
        env = detect_env()
        print(f"[nyload] env={env} (auto-detected, override with env=...)")
        args = [*args, f"env={env}"]
    sys.argv = [prog, *args]  # Hydra parses sys.argv
    entry()


def doctor() -> None:
    import torch

    from nyload.data.build import forecast_path, processed_path

    print(f"python   {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"torch    {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"cuda     {torch.version.cuda}  {props.name}  {props.total_memory / 2**30:.1f} GB  "
              f"bf16={torch.cuda.is_bf16_supported()}")
    else:
        hint = " (CPU-only torch build? see README setup)" if platform.system() == "Windows" else ""
        print(f"cuda     not available{hint}")
    print(f"env      {detect_env()} (used by train/eval unless env=... is given)")
    print(f"data     {paths.data_dir()}")
    years = [y for y in range(2000, 2100) if processed_path("CAPITL", y).exists()]
    fc = [y for y in years if forecast_path("CAPITL", y).exists()]
    print(f"         processed years: {_ranges(years) or 'none -> run `nyload data --years 2013-2023`'}")
    print(f"         NYISO forecast years: {_ranges(fc) or 'none'}")
    print(f"outputs  {paths.output_dir()}")


def _ranges(years: list[int]) -> str:
    out, start = [], None
    for i, y in enumerate(years):
        start = y if start is None else start
        if i + 1 == len(years) or years[i + 1] != y + 1:
            out.append(str(y) if start == y else f"{start}-{y}")
            start = None
    return ",".join(out)


def runs() -> None:
    root = paths.output_dir() / "runs"
    dirs = sorted(root.glob("*/metrics.json")) if root.exists() else []
    if not dirs:
        print(f"no finished runs in {root}")
        return
    print(f"{'run':<40} {'MAE':>8} {'MAPE%':>7} {'R2':>7}   NYISO MAPE%")
    for m in dirs:
        test = json.loads(m.read_text()).get("test", {})
        model, nyiso = test.get("model (hourly)", {}), test.get("NYISO forecast (hourly)", {})
        if not model:
            continue
        print(f"{m.parent.name:<40} {model['mae']:>8.1f} {model['mape']:>7.2f} {model['r2']:>7.3f}   "
              f"{nyiso.get('mape', float('nan')):.2f}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if not argv or argv[0] in ("-h", "--help") else 2)
    cmd, args = argv[0], argv[1:]
    paths.load_dotenv()

    if cmd == "doctor":
        doctor()
    elif cmd == "data":
        from nyload.data.cli import main as data_main
        if args and args[0] not in ("download", "build", "all") and args[0] not in ("-h", "--help"):
            args = ["all", *args]  # `nyload data --years ...` == download + build
        data_main(args)
    elif cmd == "train":
        from nyload.train.run import main as train_main
        _hydra(train_main, args, "nyload train")
    elif cmd == "eval":
        from nyload.eval.run import main as eval_main
        _hydra(eval_main, args, "nyload eval")
    elif cmd == "runs":
        runs()
    elif cmd == "test":
        import pytest
        sys.exit(pytest.main(["-q", str(paths.ROOT / "tests"), *args]))


if __name__ == "__main__":
    main()
