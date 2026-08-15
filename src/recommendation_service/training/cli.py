from __future__ import annotations

import argparse
import json

from .dataset import load_csv, time_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ranker Arm A, B or C")
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    train, validation = time_split(load_csv(args.input))

    if args.arm == "A":
        from .baseline import train_arm_a
        from .evaluation import binary_metrics
        from .registry import write_arm_a_artifact
        model = train_arm_a(train)
        metrics = binary_metrics(validation, model.predict)
        path = write_arm_a_artifact(args.output, args.version, model, metrics)
    elif args.arm == "B":
        from .arm_b import train_arm_b, write_arm_b_artifact
        model, calibrator, metrics = train_arm_b(train, validation)
        path = write_arm_b_artifact(args.output, args.version, model, calibrator, metrics)
    else:
        from .arm_c import train_arm_c, write_arm_c_artifact
        model, config, metrics = train_arm_c(train, validation)
        path = write_arm_c_artifact(args.output, args.version, model, config, metrics)
    print(json.dumps({"artifact": str(path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
