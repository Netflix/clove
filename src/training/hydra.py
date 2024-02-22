"""This module allows using Hydra to run the training script with minor changes to the rest of the code.

It accomplishes so by paring the existing argparse Parser options into Hydra (e.g., which can be validated), accepting
arguments as a Hydra config, and then converting them to valid argparse arguments.
"""
import argparse
import json
import logging
import multiprocessing
import os
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, make_dataclass
from enum import Enum
from typing import Any, Union

import hydra
import psutil
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.launcher import LaunchConfig, launch_agent

from training.params import ParseKwargs, create_parser


@dataclass
class SerialLaunchConfig:
    method: str = "serial"


@dataclass
class ChildProcessLaunchConfig:
    method: str = "child_process"


# Defaults from `torch.distributed.run` and:
# https://github.com/facebookresearch/hydra/blob/a53b320/contrib/hydra_torchrun_launcher/hydra_plugins/hydra_torchrun_launcher/config.py
@dataclass
class TorchRunLaunchConfig:
    _target_: str = "torch.distributed.launcher.LaunchConfig"
    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 1
    run_id: str = "none"
    role: str = "default"
    rdzv_endpoint: str = "127.0.0.1:29500"
    rdzv_backend: str = "static"
    rdzv_configs: dict[str, Any] = field(default_factory=lambda: {"rank": 0})
    rdzv_timeout: int = -1
    max_restarts: int = 0
    monitor_interval: int = 5
    # start_method: Works only with fork.
    # "spawn" and "forkserver" require pickling which doesn't work inside wrapped function
    start_method: str = "fork"
    log_dir: str | None = None
    redirects: Std = Std.NONE
    tee: Std = Std.NONE
    metrics_cfg: dict[str, str] = field(default_factory=dict)


ConfigStore.instance().store(group="internal_launcher", name="serial", node=SerialLaunchConfig)
ConfigStore.instance().store(group="internal_launcher", name="child_process", node=ChildProcessLaunchConfig)
ConfigStore.instance().store(group="internal_launcher", name="torchrun", node=TorchRunLaunchConfig)


def _argparse_parser_to_hydra_structured_config(parser: argparse.ArgumentParser, extra_fields: Sequence = ()) -> type:
    extra_field_names = {name for name, _, _ in extra_fields}

    fields = list(extra_fields)
    for action in parser._actions:
        if not isinstance(action, argparse._HelpAction):  # noqa
            name = action.option_strings[-1].lstrip("-").replace("-", "_")

            if name not in extra_field_names:
                default = action.default

                default_type = None if action.default is argparse.SUPPRESS else type(default)

                field_kwargs = {}

                if isinstance(action, ParseKwargs):
                    assigned_type = dict[str, Any]  # Note `Mapping` is not supported by Structured Configs.
                    default_type = assigned_type
                elif (isinstance(action.nargs, int) and action.nargs > 1) or action.nargs in {"*", "+"}:
                    assigned_type = list[action.type or str]
                    default_type = assigned_type if default_type is list else default_type
                elif isinstance(action, argparse._StoreConstAction):  # noqa
                    assigned_type = type(action.const)
                    field_kwargs["metadata"] = {"store_const": True, "const": action.const}

                    if (isinstance(action, argparse._StoreFalseAction)  # noqa
                            and name.startswith("no_")
                            and action.dest == name[len("no_"):]):
                        name = name.removeprefix("no_")
                        field_kwargs["metadata"]["prepend_the_word_no"] = True
                elif action.choices:
                    # Note `Literal` is not supported by Structured Configs.
                    assigned_type = Enum(name.capitalize(), action.choices)
                    default_type = assigned_type

                    if default not in {argparse.SUPPRESS, None}:
                        default = assigned_type[default]
                elif action.type is type or action.type is None:
                    assigned_type = action.type or str
                else:
                    # Don't need to use the type. Leave it as it is, then it's going to be converted by `argparse`.
                    assigned_type = str

                allowed_types = {assigned_type, default_type}
                type_ = next(iter(allowed_types)) if len(allowed_types) == 1 else Union[tuple(allowed_types)]

                if default is not argparse.SUPPRESS:
                    if isinstance(default, (list, dict, set)):  # If it's mutable. Otherwise, it raises an exception.
                        # Set `x=y` to make the variable be bound to the correct value (as opposed to the last set
                        # `default` value in the loop). See https://stackoverflow.com/a/3431699/1165181
                        field_kwargs["default_factory"] = lambda d=default: d  # noqa
                    else:
                        field_kwargs["default"] = default

                field_ = (name, type_, field(**field_kwargs))
                fields.append(field_)

    return make_dataclass("Config", fields=fields)


PARSER = create_parser()
EXTRA_FIELDS = [
    ("defaults", list, field(default_factory=lambda: [
        "_self_",
        {"hparam_search": None},
        {"launch_on": None},
        {"internal_launcher": "serial"},
        {"override hydra/job_logging": "none"},
        {"override hydra/hydra_logging": "none"},
    ])),
    ("optimized_metric_name", str, field(default="val/clip_val_loss")),
    ("internal_launcher", Any, field(default=None)),
    ("kill_all_descendent_processes_at_the_end", bool, field(default=False)),
]

OmegaConf.register_new_resolver("enum_to_name", lambda e: e.name)

Config = _argparse_parser_to_hydra_structured_config(parser=PARSER, extra_fields=EXTRA_FIELDS)
ConfigStore.instance().store(name="config", node=Config)

PARSER_FIELD_NAMES = {a.option_strings[-1] for a in PARSER._actions}

CONFIG_FIELDS_BY_NAME = {f.name: f for f in fields(Config)}  # noqa


def _cfg_to_args(cfg: DictConfig) -> Sequence[str]:
    cfg = OmegaConf.to_container(cfg, resolve=True)

    args = []

    for k, v in cfg.items():
        field = CONFIG_FIELDS_BY_NAME[k]

        if ((field.default is MISSING or v != field.default)
                and (field.default_factory is MISSING or v != field.default_factory())):
            arg_k = "--" + k.replace("_", "-")

            metadata = field.metadata

            if metadata.get("store_const") and metadata.get("prepend_the_word_no"):
                arg_k = "--no-" + arg_k.removeprefix("--")

            if arg_k in PARSER_FIELD_NAMES:
                if isinstance(v, dict):
                    flag = [arg_k, *(f"{v_k}={v_v}" for v_k, v_v in v.items())]
                elif isinstance(v, list):
                    flag = [arg_k, *(str(v_v) for v_v in v)]
                elif metadata.get("store_const"):
                    if v == metadata["const"]:
                        flag = [arg_k]
                    else:
                        raise ValueError(f"Unexpected value for {k}: {v}")
                elif isinstance(v, Enum):
                    flag = [arg_k, v.name]
                else:
                    flag = [arg_k, str(v)]

                args.extend(flag)

    return args


def _kill_all_descendent_processes() -> None:
    current_process = psutil.Process(os.getpid())

    logging.info("Terminating all descendent processes…")

    for descendent_process in current_process.children(recursive=True):
        descendent_process.terminate()

    TIMEOUT_SECONDS = 5

    for alive_process in psutil.wait_procs(current_process.children(recursive=True), timeout=TIMEOUT_SECONDS)[1]:
        logging.info(f"Killing the process {alive_process} because it didn't finish within {TIMEOUT_SECONDS}s…")
        alive_process.kill()

    psutil.wait_procs(current_process.children(recursive=True), timeout=10 * TIMEOUT_SECONDS)


def _main(args: Sequence[str]) -> Mapping[str, float]:
    from training import __main__
    return __main__.main(args)


_RESULT = {}


def _child_process_main(args: Sequence[str]) -> None:
    _RESULT = _main(args)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    # Can't set the following as part of the `extra_fields` because the default would change, and then the conversion
    # to argparse would be wrong then.
    cfg.sweep_job_id = "${hydra:job.num}"

    args = _cfg_to_args(cfg)

    launch_config = hydra.utils.instantiate(cfg.internal_launcher)

    try:
        if isinstance(launch_config, LaunchConfig):
            result = launch_agent(launch_config, _main, [args]).get(0, float("NaN"))  # noqa
        elif launch_config.method == "serial":
            result = _main(args)  # This works fine in a sweep because we kill all descendant processes at the end.
        elif launch_config.method == "child_process":
            # Using the GPU from scratch again is sometimes hard if you keep the process alive. One way to tackle this
            # is to create a new process for each Hydra run, in case this is running in multi-run mode (e.g., a sweep).
            # See:
            # * https://discuss.pytorch.org/t/clearing-the-gpu-is-a-headache/84762/10
            # * https://github.com/facebookresearch/hydra/issues/2187
            # * https://github.com/facebookresearch/hydra/issues/1331#issuecomment-1396688664
            process = multiprocessing.Process(target=_child_process_main, args=(args,), name="HydraMain")
            process.start()
            process.join()  # We wait for the process to finish to make sure all resources were freed.
            result = _RESULT
        else:
            raise ValueError(f"Unexpected internal launch config: {launch_config}")
    finally:
        if cfg.kill_all_descendent_processes_at_the_end:
            _kill_all_descendent_processes()

    result = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

    print("Results:", json.dumps(result))

    return result.get(cfg.optimized_metric_name, float("NaN"))


if __name__ == "__main__":
    main()
