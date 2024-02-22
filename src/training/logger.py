import logging
import socket
from typing import Literal

StringLogLevel = Literal["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"]


def setup_logging(level: int | StringLogLevel, log_path: str | None = None, include_host: bool = False,
                  date_format: str = "%Y-%m-%d,%H:%M:%S") -> None:
    format_elements = ["%(asctime)s", "%(levelname)s", "%(message)s"]

    if include_host:
        format_elements.insert(1, socket.gethostname())

    formatter = logging.Formatter(" | ".join(format_elements), datefmt=date_format)

    logging.root.setLevel(level)

    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(level)

    # Otherwise, it's too invasive when doing map-like operations with Datasets.
    logging.getLogger("dill").setLevel(logging.WARNING)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
