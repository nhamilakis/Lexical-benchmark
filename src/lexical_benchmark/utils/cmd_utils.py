import abc
import os
from pathlib import Path

os.environ["STELA_VERSION"] = "3"

import argparse
import inspect
import typing as t
from dataclasses import dataclass


@dataclass
class CommandMetadata:
    """Metadata for a command.

    Stores information about a command's name, description, and method reference.

    Raises:
        None

    """

    name: str
    description: str
    method: t.Callable


class CommandRunnerCLI(abc.ABC):
    """CLI tool for running multiple commands."""

    @abc.abstractmethod
    def description() -> str:
        """CMD description."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description=self.description())
        self._setup_parser()

    def _setup_parser(self) -> None:
        """Set up the argument parser with subparsers for each command."""
        subparsers = self.parser.add_subparsers(dest="command", help="Command to execute")

        # Register each command as a subparser
        for cmd in self._get_commands():
            cmd_parser = subparsers.add_parser(cmd.name, help=cmd.description)
            # Inspect the method signature to add arguments automatically
            self._add_arguments_from_method(cmd_parser, cmd.method)

    def _get_commands(self) -> list[CommandMetadata]:
        """Get all command methods in this class.

        Identifies methods prefixed with 'cmd_' and extracts their metadata.

        Raises:
            None

        """
        commands = []

        for name, method in inspect.getmembers(self, inspect.ismethod):
            if name.startswith("cmd_"):
                # Extract command name (remove cmd_ prefix)
                cmd_name = name[4:]
                # Get the docstring for description
                description = method.__doc__.split("\n")[0] if method.__doc__ else ""
                commands.append(CommandMetadata(cmd_name, description, method))

        return commands

    def _add_arguments_from_method(self, parser: argparse.ArgumentParser, method: t.Callable) -> None:
        """Add arguments to a parser based on a method's parameters.

        Uses type annotations to determine argument types and default values.

        Raises:
            None

        """
        sig = inspect.signature(method)

        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter
            if param_name == "self":
                continue

            # Get type annotation and default value
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            default = param.default if param.default != inspect.Parameter.empty else None
            required = param.default == inspect.Parameter.empty

            # Handle special types
            if param_type == Path:
                param_type = str  # argparse doesn't handle Path directly

            # Add the argument
            parser.add_argument(
                f"--{param_name}", type=param_type, default=default, required=required, help=f"{param_name} parameter"
            )

    def run(self, args: list[str] | None = None) -> None:
        """Run the CLI with the given arguments.

        Raises:
            AttributeError: If an invalid command is specified

        """
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return

        # Get the method to call
        method_name = f"cmd_{parsed_args.command}"
        if not hasattr(self, method_name):
            self.parser.error(f"Unknown command: {parsed_args.command}")

        # Call the method with the parsed arguments
        method = getattr(self, method_name)

        # Convert the namespace to a dictionary and remove the command
        args_dict = vars(parsed_args)
        args_dict.pop("command")

        # Call the method with the arguments
        method(**args_dict)
