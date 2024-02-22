#!/usr/bin/env python
import os
import sys

import yaml


def main() -> None:
    if len(sys.argv) > 2:
        raise ValueError("Too many arguments. Should be at most one argument for a filename.")

    with (sys.stdin if len(sys.argv) == 1 or sys.argv[1] == "-" else open(sys.argv[1])) as file:
        runtime_env = yaml.safe_load(file)

    runtime_env.setdefault("env_vars", {})
    for k in ["HUGGING_FACE_HUB_TOKEN", "NEPTUNE_API_TOKEN"]:
        runtime_env["env_vars"][k] = os.environ[k]

    print(yaml.dump(runtime_env))


if __name__ == "__main__":
    main()
