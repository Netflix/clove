import argparse
from typing import Any


# Adapted from https://github.com/allenai/allennlp/blob/3aafb92/allennlp/commands/__init__.py
class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """Custom argument parser that will display the default value for an argument in the help message. """

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default: Any) -> bool:
        return default is None or (isinstance(default, (str, list, tuple, set)) and not default)

    def add_argument(self, *args, **kwargs) -> argparse.Action:
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get("action") not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            kwargs["help"] = f"{kwargs.get('help', '')} (default = {default})"
        return super().add_argument(*args, **kwargs)
