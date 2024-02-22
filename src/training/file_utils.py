import logging
import multiprocessing
import subprocess
import time
from typing import Any, Literal

import fsspec
import torch
from cached_path import cached_path
from torch.serialization import MAP_LOCATION
from torch.utils.data._utils.worker import ManagerWatchdog

Protocol = Literal["fsspec", "s3"]


def remote_sync_s3(local_dir: str, remote_dir: str) -> bool:
    # Skip "epoch_latest.pt", which can change during the sync.
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, "--exclude", "*epoch_latest.pt"],
                            capture_output=True)
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with the S3 bucket: {result.stderr.decode('utf-8')}")
        return False

    logging.info(f"Successfully synced with the S3 bucket.")
    return True


def remote_sync_fsspec(local_dir: str, remote_dir: str) -> bool:
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        if "epoch_latest.pt" in k:  # Skip this file, which can change during the sync.
            continue

        logging.info(f"Attempting to sync {k}")
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f"Skipping remote sync for {k}.")
            continue

        try:
            logging.info(f"Successful sync for {k}.")
            b[k] = a[k]
        except Exception as e:
            logging.info(f"Error during remote sync for {k}: {e}")
            return False

    return True


def remote_sync(local_dir: str, remote_dir: str, protocol: Protocol) -> bool:
    logging.info("Starting remote sync.")
    if protocol == "s3":
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == "fsspec":
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error(f"Remote protocol not known: {protocol}")
        return False


def _keep_running_remote_sync(sync_every: int, local_dir: str, remote_dir: str, protocol: Protocol,
                              check_watchdog_every: int = 5) -> None:
    # Chances are that this subprocess is never killed. It has happened, probably because of some bug of Ray or GRPC.
    # So we check if the parent process is still alive.
    watchdog = ManagerWatchdog()

    time.sleep(sync_every)

    while watchdog.is_alive():
        remote_sync(local_dir, remote_dir, protocol)

        # Sleep for `sync_every` seconds in total, but check if the parent process is still alive every
        # `check_watchdog_every` to free the underlying resources faster (e.g., the GPUs).
        time.sleep(sync_every % check_watchdog_every)
        for _ in range(sync_every // check_watchdog_every):
            if not watchdog.is_alive():
                break
            time.sleep(check_watchdog_every)


def start_sync_process(sync_every: int, local_dir: str, remote_dir: str, protocol: Protocol) -> multiprocessing.Process:
    return multiprocessing.Process(target=_keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol),
                                   name="RemoteSync", daemon=True)


def pt_load(path_or_url: str, map_location: MAP_LOCATION = "cpu") -> Any:
    with open(cached_path(path_or_url), "rb") as f:
        return torch.load(f, map_location=map_location)
