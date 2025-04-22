#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pip Package Updater
============================

This module provides a class `PipUpdater` to check for and install updates
for installed pip packages, excluding those whose latest available version
is a source distribution (sdist).
"""

import sys

REQUIRED_PYTHON_VERSION = (3, 11)

current_version = sys.version_info

if current_version < REQUIRED_PYTHON_VERSION:
    print(
        f"Error: This script requires Python version {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} or later."
    )
    print(
        f"You are using Python {current_version.major}.{current_version.minor}.{current_version.micro}."
    )
    sys.exit(1)

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import time
import traceback

from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Self,
    Tuple,
    TypeVar,
    Union,
    cast,
)

try:
    from loguru import logger
    from rich.console import Console
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        DownloadColumn,
        TransferSpeedColumn,  # Added for potential future use / display
    )
    from packaging.version import parse as parse_version, InvalidVersion, Version
    from packaging.specifiers import SpecifierSet, Specifier, InvalidSpecifier
    import diskcache  # For caching outdated check results
except ImportError as e:
    print(
        f"Error: Missing required libraries ({e.name}). Please install them: pip install loguru rich packaging diskcache"
    )
    sys.exit(1)

# --- Constants ---
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pip_updater"
DEFAULT_CACHE_DURATION_MINUTES = 60  # Default cache validity: 1 hour
OUTDATED_CACHE_TAG: Final[str] = "outdated"

# --- Custom Exceptions ---


class PipUpdaterError(Exception):
    """Base exception for the PipUpdater class."""

    pass


class PipCommandError(PipUpdaterError):
    """Raised when a pip command fails unexpectedly."""

    def __init__(self, command: str, stderr: str, return_code: int):
        self.command = command
        self.stderr = stderr
        self.return_code = return_code
        super().__init__(
            f"Pip command '{command}' failed with code {return_code}:\n{stderr}"
        )


class PackageUpdateError(PipUpdaterError):
    """Raised when a specific package fails to update."""

    def __init__(self, package_name: str, reason: str):
        self.package_name = package_name
        self.reason = reason
        super().__init__(f"Failed to update package '{package_name}': {reason}")


class CacheError(PipUpdaterError):
    """Raised for issues related to caching."""

    pass


class CachePruningError(CacheError):
    """Raised for issues specifically during cache pruning."""

    def __init__(self, package_name: Optional[str], details: Union[str, Exception]):
        self.package_name = package_name
        self.details = details
        pkg_info = f" for '{package_name}'" if package_name else ""
        super().__init__(f"Cache pruning failed{pkg_info}: {details}")


class DependencyAnalysisError(PipUpdaterError):
    """Base exception for errors in this dependency analysis module."""

    pass


class EnvironmentStateError(DependencyAnalysisError):
    """Exception raised when an operation is attempted before the environment state is loaded."""

    pass


class PipdeptreeExecutionError(DependencyAnalysisError):
    """Exception raised when the pipdeptree subprocess fails."""

    def __init__(self, return_code: int, stderr: str):
        self.return_code = return_code
        self.stderr = stderr
        super().__init__(
            f"pipdeptree command failed with exit code {return_code}.\nStderr: {stderr}"
        )


class PipdeptreeOutputError(DependencyAnalysisError):
    """Exception raised when pipdeptree output is malformed or unexpected."""

    pass


class InvalidConstraintError(DependencyAnalysisError):
    """Exception raised when a package version or specifier string is invalid."""

    def __init__(
        self,
        package_name: str,
        constraint: str,
        context: str,
        original_exception: Exception,
    ):
        self.package_name = package_name
        self.constraint = constraint
        self.context = context  # e.g., 'proposed version', 'required specifier'
        self.original_exception = original_exception
        super().__init__(
            f"Failed to parse {context} '{constraint}' for package '{package_name}'. "
            f"Original error: {original_exception}"
        )


# --- Data Structures ---


@dataclass(frozen=True, kw_only=True)
class PackageInfo:
    """Represents information about an installed package."""

    name: str
    version: str
    location: Optional[str] = None
    installer: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class OutdatedPackageInfo(PackageInfo):
    """Represents information about an outdated package."""

    latest_version: str
    latest_filetype: str  # e.g., 'wheel', 'sdist'


@dataclass
class UpdateStats:
    """Stores statistics about the update process."""

    checked_count: int = 0
    outdated_count: int = 0
    sdist_count: int = 0
    excluded_count: int = 0
    attempted_update_count: int = 0
    successful_update_count: int = 0
    failed_update_count: int = 0
    conflict_detected_count: int = 0
    conflict_ignored_count: int = 0
    system_managed_count: int = 0
    packages_to_update: List[str] = field(default_factory=list)
    updated_packages: Dict[str, Tuple[str, str]] = field(
        default_factory=dict
    )  # name: (old_v, new_v)
    failed_packages: Dict[str, str] = field(default_factory=dict)  # name: reason
    conflicted_packages: List[str] = field(
        default_factory=list
    )  # Packages skipped due to conflict (when not ignored)
    system_packages: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_download_size_bytes: int = 0
    cache_used: bool = False

    @property
    def duration(self) -> Optional[float]:
        """Calculates the duration of the update process in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def total_download_size_mb(self) -> float:
        """Returns the total download size in megabytes."""
        return self.total_download_size_bytes / (1024 * 1024)


# Define specific type alias for the structure of a package entry
# from pipdeptree --json-tree output for better readability.
# Note: 'installed_version' and 'required_version' can sometimes be empty strings or None.
# 'key' is the canonical package name used in dependencies.
# Detailed structure analysis from pipdeptree output:
# Each item in the list is a dict representing a package:
# {
#   "package": {
#     "key": str,
#     "package_name": str,
#     "installed_version": str,
#     "required_version": Optional[str], # The version required *of this package* if it's a dependency
#     "installer": Optional[str]
#   },
#   "dependencies": [ # List of packages this package depends on
#     {
#       "key": str, # The canonical name of the dependency
#       "package_name": str,
#       "installed_version": str, # The version installed *of the dependency*
#       "required_version": Optional[str] # The version specifier *this package* requires of the dependency
#     },
#     ...
#   ]
# }

PipdeptreePackageInfo = Dict[str, str | None]
PipdeptreeDependencyInfo = Dict[str, str | None]
PipdeptreeEntry = Dict[str, PipdeptreePackageInfo | List[PipdeptreeDependencyInfo]]
# Define the type for the reverse dependency map
# Maps package name (str) to a list of tuples:
# (depender_package_name: str, required_version_spec: str)
ReverseDependencyMap = Dict[str, List[Tuple[str, str]]]
# Define a generic type for the method being decorated
F = TypeVar("F", bound=Callable[..., Any])


# --- Time Operations ---
@contextlib.contextmanager
def timed_block(name: Optional[str] = "Updater"):
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.debug(
            f"{name} completed in {format_duration(time.perf_counter() - start)}"
        )


def format_duration(seconds: int) -> str:
    """
    Converts a duration in seconds into a human-readable string using the most appropriate time unit.

    Args:
        seconds (int): The total duration in seconds.

    Returns:
        str: A human-friendly string representation of the duration.
    """
    SECONDS_PER_MINUTE: Final = 60
    SECONDS_PER_HOUR: Final = 3600
    SECONDS_PER_DAY: Final = 86400

    if seconds < SECONDS_PER_MINUTE:
        value = seconds
        unit = "second"
    elif seconds < SECONDS_PER_HOUR:
        value = seconds / SECONDS_PER_MINUTE
        unit = "minute"
    elif seconds < SECONDS_PER_DAY:
        value = seconds / SECONDS_PER_HOUR
        unit = "hour"
    else:
        value = seconds / SECONDS_PER_DAY
        unit = "day"

    display_value = int(value) if value.is_integer() else int(round(value, 0))
    plural = "s" if display_value != 1 else ""

    return f"{display_value} {unit}{plural}"


# --- Dependency Analyzer Class ---


class DependencyAnalyzer:
    """
    Manages the dependency state of the current Python environment
    and provides functionality to check for version conflicts.

    The dependency state (pipdeptree output and reverse map) is loaded
    explicitly using the 'load_current_environment_state' method.
    """

    def __init__(self, ignore_conflicts: bool = False):
        """Initializes the analyzer with no state loaded."""
        self._dependency_tree: Optional[List[PipdeptreeEntry]] = None
        self._reverse_map: Optional[ReverseDependencyMap] = None
        self.ignore_conflicts = ignore_conflicts
        logger.debug("DependencyAnalyzer instance created. State is not yet loaded.")

    def load_current_environment_state(self):
        """
        Loads the current dependency state of the Python environment
        by executing pipdeptree and building the internal maps.

        This method should be called after any potential change to the
        environment's installed packages (install, upgrade, uninstall).

        Raises:
            PipdeptreeExecutionError: If the pipdeptree subprocess fails.
            PipdeptreeOutputError: If pipdeptree output is malformed.
            DependencyAnalysisError: For other unexpected errors during loading.
        """
        logger.info("Loading current environment dependency state...")
        try:
            self._dependency_tree = self._get_pipdeptree_json()
            logger.debug(
                f"Successfully retrieved dependency tree with {len(self._dependency_tree)} packages."
            )

            self._reverse_map = self._build_reverse_dependency_map(
                self._dependency_tree
            )
            logger.debug(
                f"Successfully built reverse dependency map with {len(self._reverse_map)} entries."
            )

        except Exception:
            # Ensure state is cleared if loading fails
            self._dependency_tree = None
            self._reverse_map = None
            # Re-raise the caught exception
            raise

    @staticmethod
    def refresh_dependency_tree(
        extra_func: Callable[[Any], None], flag_name: str = "ignore_conflicts"
    ) -> Callable[[F], F]:
        """
        Decorator that runs `extra_func(self)` if `self.<flag_name>` is True.

        Args:
            extra_func: A method that takes `self` as its only argument.
            flag_name: The name of the instance attribute that acts as a boolean flag.
        Returns:
            The wrapped function, unchanged if flag is False, with side effect if True.
        """

        def decorator(method: F) -> F:
            @wraps(method)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self = args[0]
                if not getattr(self, flag_name, False):
                    logger.debug("Refreshing dependency tree after processing package")
                    extra_func(self)
                return method(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    @refresh_dependency_tree(
        load_current_environment_state, flag_name="ignore_conflicts"
    )
    def check_package_conflict(
        self, package_name: str, proposed_version: str
    ) -> List[str]:
        """
        Checks for potential version conflicts if a given package's version
        were updated to `proposed_version`, based on the *currently loaded*
        environment state.

        Args:
            package_name: The name of the package being checked (e.g., 'requests').
            proposed_version: The proposed new version string for the package
                              (e.g., '2.28.1').

        Returns:
            A list of strings. Each string describes a violation detected.
            Returns an empty list if no conflicts are found or if the package
            has no known dependers in the currently loaded state.

        Raises:
            EnvironmentStateError: If `load_current_environment_state` has not
                                   been called successfully.
            ValueError: If package_name or proposed_version are empty strings.
            InvalidConstraintError: If proposed_version cannot be parsed.
            DependencyAnalysisError: For unexpected errors during constraint checking.
        """
        if self._reverse_map is None:
            raise EnvironmentStateError(
                "Dependency state is not loaded. Call 'load_current_environment_state' first."
            )

        if not package_name:
            raise ValueError("package_name cannot be an empty string.")
        if not proposed_version:
            raise ValueError("proposed_version cannot be an empty string.")

        logger.debug(
            f"Checking conflicts for package '{package_name}' with proposed version '{proposed_version}' (using loaded state)."
        )

        violations: List[str] = []

        # Check if the package exists as a key in the reverse dependency map
        if package_name not in self._reverse_map:
            logger.info(
                f"Package '{package_name}' has no known dependers in the loaded state. No conflicts found."
            )
            return violations  # No dependers means no potential conflicts

        dependers = self._reverse_map[package_name]
        logger.debug(f"Found {len(dependers)} dependers for '{package_name}'.")

        try:
            # Attempt to parse the proposed version string once
            parsed_proposed_version: Version = parse_version(proposed_version)
            logger.debug(
                f"Successfully parsed proposed version: {parsed_proposed_version}"
            )
        except InvalidVersion as e:
            # Catch specific packaging error for invalid version strings
            logger.error(
                f"Failed to parse proposed version '{proposed_version}' for '{package_name}': {e}",
                exc_info=True,
            )
            raise InvalidConstraintError(
                package_name, proposed_version, "proposed version", e
            ) from e
        except Exception as e:
            # Catch any other unexpected error during version parsing
            logger.error(
                f"An unexpected error occurred while parsing proposed version '{proposed_version}' for '{package_name}': {e}",
                exc_info=True,
            )
            raise DependencyAnalysisError(
                f"Unexpected error parsing proposed version: {e}"
            ) from e

        for depender_name, required_spec_str in dependers:
            # Skip if the required specifier is empty or explicitly "any"
            if not required_spec_str or required_spec_str.strip().lower() == "any":
                logger.debug(
                    f"Depender '{depender_name}' requires '{package_name}' with specifier '{required_spec_str}'. Skipping check."
                )
                continue

            logger.debug(
                f"Checking depender '{depender_name}' with requirement '{package_name}{required_spec_str}'."
            )

            try:
                # Parse the required version specifier string
                required_specifier: SpecifierSet = SpecifierSet(required_spec_str)

                # Check if the proposed new version is compatible with the specifier
                if parsed_proposed_version not in required_specifier:
                    violation_message = (
                        f"'{depender_name}' requires "
                        f"'{package_name}{required_spec_str}', which is incompatible with proposed version '{proposed_version}'."
                    )
                    logger.warning(violation_message)
                    violations.append(violation_message)
                else:
                    logger.debug(
                        f"Proposed version '{proposed_version}' is compatible with '{depender_name}' requirement '{required_spec_str}'."
                    )

            except InvalidSpecifier as e:
                # Catch specific packaging error for invalid specifier strings
                error_message = (
                    f"Invalid version specifier '{required_spec_str}' for package '{package_name}' "
                    f"required by '{depender_name}'. Original error: {e}"
                )
                logger.error(error_message, exc_info=True)
                violations.append(
                    f"Constraint parsing error: {depender_name} requires {package_name}{required_spec_str} (Invalid Specifier: {e})"
                )
            except Exception as e:
                # Catch any other unexpected error during specifier parsing or evaluation
                error_message = (
                    f"An unexpected error occurred evaluating version specifier '{required_spec_str}' "
                    f"for package '{package_name}' required by '{depender_name}'. Original error: {e}"
                )
                logger.error(error_message, exc_info=True)
                violations.append(
                    f"Constraint evaluation error: {depender_name} requires {package_name}{required_spec_str} (Error: {e})"
                )

        if violations:
            logger.warning(
                f"Found {len(violations)} version conflicts for '{package_name}' at version '{proposed_version}'."
            )
        else:
            logger.info(
                f"No version conflicts found for '{package_name}' at version '{proposed_version}'."
            )

        return violations

    def __repr__(self) -> str:
        """
        Provides a developer-friendly string representation of the analyzer instance.
        Includes information about the loaded state and counts.
        """
        state = "Loaded" if self else "Not Loaded"
        if self:
            num_packages = len(self)
            num_reverse_deps = (
                len(self._reverse_map) if self._reverse_map is not None else 0
            )
            return (
                f"<DependencyAnalyzer(state='{state}', packages={num_packages}, "
                f"reverse_deps={num_reverse_deps}, id={id(self)})>"
            )
        else:
            return f"<DependencyAnalyzer(state='{state}', id={id(self)})>"

    def __bool__(self) -> bool:
        """
        Allows boolean evaluation of the analyzer.
        An analyzer is considered 'True' if the dependency state has been successfully loaded.
        """
        return self._reverse_map is not None

    def __len__(self) -> int:
        """
        Allows using the len() function on the analyzer instance.
        Returns the number of packages in the dependency tree if loaded, otherwise 0.
        """
        return len(self._dependency_tree) if self._dependency_tree is not None else 0

    def _get_pipdeptree_json(self) -> List[PipdeptreeEntry]:
        """
        Internal helper: Executes 'pipdeptree --json-tree' and parses output.
        Handles subprocess execution and JSON parsing errors.
        """
        command = [sys.executable, "-m", "pipdeptree", "--json-tree", "--local-only"]
        logger.info(f"Executing command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, encoding="utf-8"
            )
            logger.debug(f"Command finished with return code: {result.returncode}")

        except FileNotFoundError:
            logger.error(
                f"Command not found: {' '.join(command[:2])}. Is pipdeptree installed in the current environment?"
            )
            raise PipdeptreeExecutionError(
                127, f"Executable not found: {command[0]} {command[1]}"
            )
        except PermissionError:
            logger.error(
                f"Permission denied when trying to execute: {' '.join(command)}. Check file permissions."
            )
            raise PipdeptreeExecutionError(
                126, f"Permission denied: {' '.join(command)}"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during subprocess execution: {e}",
                exc_info=True,
            )
            raise DependencyAnalysisError(f"Unexpected subprocess error: {e}") from e

        if result.returncode != 0:
            logger.error(f"pipdeptree command failed. Stderr: {result.stderr}")
            raise PipdeptreeExecutionError(result.returncode, result.stderr)

        if not result.stdout.strip():
            logger.warning("'pipdeptree' command produced no output.")
            return []

        try:
            data = json.loads(result.stdout)
            if not isinstance(data, list):
                logger.error(
                    f"'pipdeptree' output is not a JSON list. Type: {type(data)}"
                )
                raise PipdeptreeOutputError(
                    f"Expected JSON list output from pipdeptree, but received {type(data)}."
                )
            logger.debug("Successfully parsed 'pipdeptree' JSON output.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pipdeptree JSON output: {e}", exc_info=True)
            raise PipdeptreeOutputError(
                f"Failed to decode JSON output from 'pipdeptree': {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON parsing: {e}", exc_info=True
            )
            raise PipdeptreeOutputError(
                f"Unexpected error during JSON parsing: {e}"
            ) from e

    def _build_reverse_dependency_map(
        self, tree: List[PipdeptreeEntry]
    ) -> ReverseDependencyMap:
        """
        Internal helper: Constructs a reverse dependency map from pipdeptree output.
        Includes checks for data structure integrity.
        """
        reverse_deps: ReverseDependencyMap = {}
        logger.info(
            f"Building reverse dependency map from {len(tree)} package entries."
        )

        if not tree:
            logger.warning(
                "Input tree is empty. Returning an empty reverse dependency map."
            )
            return reverse_deps

        for package_entry in tree:
            if not isinstance(package_entry, dict):
                logger.warning(
                    f"Skipping non-dictionary entry in tree: {package_entry!r}"
                )
                continue

            try:
                parent_package_info = package_entry
                if not parent_package_info or not isinstance(parent_package_info, dict):
                    logger.warning(
                        f"Skipping entry with invalid or missing 'package' field: {package_entry!r}"
                    )
                    continue

                parent_name = parent_package_info.get("package_name")
                if not parent_name or not isinstance(parent_name, str):
                    logger.warning(
                        f"Skipping entry with invalid or missing 'package_name' in 'package' field: {package_entry!r}"
                    )
                    continue

                dependencies = package_entry.get("dependencies", [])

                if not isinstance(dependencies, list):
                    logger.warning(
                        f"Dependencies field is not a list for {parent_name}: {dependencies!r}. Treating as empty."
                    )
                    dependencies = []

                logger.debug(
                    f"Processing package: {parent_name} with {len(dependencies)} listed dependencies."
                )

                for dep_info in dependencies:
                    if not isinstance(dep_info, dict):
                        logger.warning(
                            f"Skipping non-dictionary dependency entry for {parent_name}: {dep_info!r}"
                        )
                        continue

                    try:
                        child_key = dep_info.get("key")
                        if not child_key or not isinstance(child_key, str):
                            logger.warning(
                                f"Skipping dependency with invalid or missing 'key' for {parent_name}: {dep_info!r}"
                            )
                            continue

                        required_spec = dep_info.get("required_version", "")
                        if required_spec is None:
                            required_spec = ""  # Ensure None is treated as empty string

                        if not isinstance(required_spec, str):
                            logger.warning(
                                f"Required version specifier is not a string for {child_key} by {parent_name}: {required_spec!r}. Treating as empty."
                            )
                            required_spec = ""

                        logger.debug(
                            f"• Dependency → '{child_key}' (Required Spec → '{required_spec}')"
                        )

                        if child_key not in reverse_deps:
                            reverse_deps[child_key] = []

                        reverse_deps[child_key].append((parent_name, required_spec))

                    except Exception as e:
                        logger.error(
                            f"Unexpected error processing dependency {dep_info!r} for package {parent_name}: {e}",
                            exc_info=True,
                        )
                        # Continue processing other dependencies
                        continue

            except Exception as e:
                logger.error(
                    f"Unexpected error processing package entry {package_entry!r}: {e}",
                    exc_info=True,
                )
                # Continue processing other package entries
                continue

        logger.debug(
            f"Finished building reverse dependency map. Found {len(reverse_deps)} dependent packages."
        )
        return reverse_deps


# --- The Main Class ---


class PipUpdater:
    """
    Checks for and installs updates for pip packages, excluding sdists,
    with caching, conflict detection, and detailed logging.

    Args:
        exclude_packages (Optional[List[str]]): Packages to always exclude.
        log_level (str): Minimum logging level.
        log_to_file (bool): Whether to log to a file.
        log_file_path (str | Path): Path for the log file.
        rich_console (bool): Use Rich for enhanced console output.
        cache_duration_minutes (int): How long to cache outdated check results (in minutes).
        ignore_conflicts (bool): If True, attempt updates even if pip detects dependency conflicts.
        cache_dir (str | Path): Directory to store cache files.
        allow_break_system_pkgs (bool): Upgrade system-managed packages with pip
    """

    def __init__(
        self: Self,
        exclude_packages: Optional[List[str]] = None,
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_file_path: str | Path = "pip_updater.log",
        rich_console: bool = True,
        cache_duration_minutes: int = DEFAULT_CACHE_DURATION_MINUTES,
        ignore_conflicts: bool = False,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        allow_break_system_pkgs: bool = False,
    ) -> None:
        self.exclude_packages: List[str] = exclude_packages or []
        self.log_level: str = log_level.upper()
        self.log_to_file: bool = log_to_file
        self.log_file_path: Path = Path(log_file_path)
        self.use_rich_console: bool = rich_console
        self.cache_duration_seconds: int = cache_duration_minutes * 60
        self.ignore_conflicts: bool = ignore_conflicts
        self.cache_dir: Path = Path(cache_dir).resolve()
        self._allow_break_system_pkgs: bool = allow_break_system_pkgs
        self.console = Console(
            stderr=True,
            record=True,
            theme=Theme(
                {
                    "logging.level.info": "bold magenta",
                }
            ),
        )
        self.cache: Optional[diskcache.Cache] = None

        self._setup_logger()
        self._setup_cache()

        self._analyzer = DependencyAnalyzer()

        self.installed_packages: List[PackageInfo] = []
        self.outdated_packages: List[OutdatedPackageInfo] = []
        self.packages_to_update: List[OutdatedPackageInfo] = []
        self.stats: UpdateStats = UpdateStats()

        logger.info("PipUpdater initialized")
        logger.debug(f"Excluded packages: {self.exclude_packages}")
        logger.debug(f"Log level: {self.log_level}")
        logger.debug(
            f"Log to file: {self.log_to_file} (Path: {self.log_file_path if self.log_to_file else 'Disabled'})"
        )
        logger.debug(f"Rich console: {self.use_rich_console}")
        logger.debug(f"Cache duration: {cache_duration_minutes} minutes")
        logger.debug(f"Cache directory: {self.cache_dir}")
        logger.debug(f"Ignore conflicts: {self.ignore_conflicts}")
        logger.debug(f"Update system packages: {self._allow_break_system_pkgs}")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Pip version: {self._get_pip_version()}")
        logger.debug(
            f"Platform: {platform.system()} {platform.release()} ({platform.machine()})"
        )
        logger.debug(f"Dependecy analyzer instance: {self._analyzer}")

    def _setup_logger(self: Self) -> None:
        """Configures the Loguru logger."""
        logger.remove()  # Remove default handler

        file_log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        stderr_log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

        # Console Handler (Rich or Standard)
        if self.use_rich_console:
            logger.add(
                RichHandler(
                    console=self.console,
                    rich_tracebacks=True,
                    markup=True,
                    show_path=True,  # Show path/line number
                    tracebacks_show_locals=True,  # Show locals in tracebacks for debugging
                ),
                level=self.log_level,
                format="{message}",
            )
        else:
            logger.add(
                sys.stderr,
                level=self.log_level,
                format=stderr_log_format,  # Use format with line number
                colorize=True,
            )

        # File Handler
        if self.log_to_file:
            try:
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.add(
                    self.log_file_path,
                    level="DEBUG",  # Log everything to file
                    format=file_log_format,  # Use format with line number
                    rotation="10 MB",
                    retention="7 days",
                    encoding="utf-8",
                )
                logger.info(f"Logging detailed output to file: {self.log_file_path}")
            except OSError as e:
                # Use basic print if logger failed but console might work
                print(
                    f"ERROR: Failed to configure file logging to {self.log_file_path}: {e}",
                    file=sys.stderr,
                )
                self.log_to_file = False  # Disable file logging if setup fails

        logger.success("Logger configured successfully.")

    def _setup_cache(self: Self) -> None:
        """Initializes the disk cache."""
        if self.cache_duration_seconds <= 0:
            logger.info("Cache duration is zero or negative, caching disabled.")
            self.cache = None
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(self.cache_dir))
            logger.info(f"Disk cache initialized at: {self.cache_dir}")
            # Prune expired items on startup
            self.cache.expire()
        except Exception as e:
            logger.error(
                f"Failed to initialize disk cache at '{self.cache_dir}': {e}. Caching will be disabled."
            )
            self.cache = None

    def _get_cache_key(self: Self) -> str:
        """Generates a cache key based on the Python environment."""
        # Key should reflect the environment to avoid cache collisions
        # Using sys.prefix which usually points to the environment root
        return f"outdated_packages_{sys.prefix}"

    def _run_pip_command(self: Self, command: List[str]) -> Tuple[str, str, int]:
        """Runs a pip command using subprocess, ensuring UTF-8 encoding."""
        full_command = [sys.executable, "-m", "pip"] + command
        command_str = " ".join(full_command)
        logger.debug(f"Executing command: {command_str}")
        try:
            process = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            logger.debug(f"Command finished with return code: {process.returncode}")

            stdout = process.stdout.strip() if process.stdout else ""
            stderr = process.stderr.strip() if process.stderr else ""

            if stdout:
                logger.trace(f"Command stdout:\n{stdout}")
            if stderr:
                log_level = "WARNING"
                is_error = process.returncode != 0
                is_conflict_warning = False

                if is_error and not is_conflict_warning:
                    log_level = "ERROR"
                elif is_error and is_conflict_warning and not self.ignore_conflicts:
                    log_level = "ERROR"

                logger.log(log_level, f"Command stderr: {stderr}")

            return stdout, stderr, process.returncode

        except FileNotFoundError:
            logger.critical(
                f"Error: '{sys.executable} -m pip' command not found. Is pip installed correctly?"
            )
            raise PipCommandError(
                command=command_str, stderr="pip command not found", return_code=-1
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while running command '{command_str}': {e}"
            )
            logger.opt(exception=True).debug(
                "Traceback for unexpected subprocess error:"
            )
            raise PipCommandError(
                command=command_str, stderr=str(e), return_code=-1
            ) from e

    def _get_pip_version(self: Self) -> str:
        """Gets the currently installed pip version."""
        try:
            stdout, _, return_code = self._run_pip_command(["--version"])
            if return_code == 0 and stdout:
                match = re.search(r"pip\s+([\d.]+)", stdout)
                if match:
                    version = match.group(1)
                    return version
                logger.warning(f"Could not parse pip version from output: {stdout}")
            return "Unknown"
        except PipCommandError as e:
            logger.error(f"Failed to get pip version: {e}")
            return "Error retrieving version"

    def _get_installed_packages(self: Self) -> List[PackageInfo]:
        """Retrieves a list of all installed packages using `pip list`."""
        logger.info("Retrieving list of installed packages...")
        command = ["list", "--format=json", "--disable-pip-version-check"]
        stdout, stderr, return_code = self._run_pip_command(command)

        if return_code != 0:
            raise PipCommandError(
                command=" ".join(command), stderr=stderr, return_code=return_code
            )

        try:
            installed_data = json.loads(stdout)
            packages = [
                PackageInfo(
                    name=pkg.get("name", "Unknown"),
                    version=pkg.get("version", "Unknown"),
                    location=pkg.get("location", "Unknown"),
                    installer=pkg.get("installer", "Unknown"),
                )
                for pkg in installed_data
            ]
            count = len(packages)
            logger.success(f"Found {count} installed packages.")
            self.stats.checked_count = count
            return packages
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from 'pip list': {e}")
            logger.debug(f"Raw stdout for 'pip list':\n{stdout}")
            raise PipUpdaterError("Could not parse list of installed packages.") from e
        except KeyError as e:
            logger.error(f"Unexpected JSON structure from 'pip list': Missing key {e}")
            logger.debug(f"Raw JSON data: {stdout}")
            raise PipUpdaterError(
                "Unexpected format for installed package list."
            ) from e

    def _get_outdated_packages(
        self: Self, current_installed_hash: Optional[str]
    ) -> List[OutdatedPackageInfo]:
        """
        Retrieves a list of outdated packages, using cache if available and valid.

        Args:
            current_installed_hash: The SHA256 hash of the current `pip list --format=json` output.
                                     Used to validate cache freshness beyond just time.
        """
        logger.info("Checking for outdated packages (using cache if available)...")
        cache_key = self._get_cache_key()
        cached_outdated_str: Optional[str] = (
            None  # Store JSON string from cache/live command
        )
        cache_status_msg = "Cache disabled or check failed."  # Default message
        self.stats.cache_used = False  # Reset cache usage flag for this specific check

        # --- Cache Check ---
        if self.cache and self.cache_duration_seconds > 0 and current_installed_hash:
            try:
                cached_value = self.cache.get(
                    cache_key, default=None, expire_time=True, tag=OUTDATED_CACHE_TAG
                )
                if cached_value is not None:
                    (
                        (stored_outdated_str, stored_installed_hash),
                        expiry_timestamp,
                        _,
                    ) = cached_value
                    current_time = time.time()

                    if expiry_timestamp > current_time:
                        # Cache is valid both time-wise and content-wise
                        if stored_installed_hash == current_installed_hash:
                            # Check type just in case something else was stored
                            if isinstance(stored_outdated_str, str):
                                expiry_dt = datetime.fromtimestamp(expiry_timestamp)
                                remaining_s = int(expiry_timestamp - current_time)
                                cache_status_msg = f"Using cached data (valid until {expiry_dt.strftime('%Y-%m-%d %H:%M:%S')}, {format_duration(remaining_s)} remaining)."
                                logger.success(cache_status_msg)
                                cached_outdated_str = stored_outdated_str
                                self.stats.cache_used = True
                            else:
                                cache_status_msg = f"Invalid data type found in cache for key '{cache_key}'. Ignoring cache."
                                logger.warning(cache_status_msg)
                                self.cache.delete(cache_key)  # Remove invalid entry
                        else:
                            # Hashes don't match - installed packages have changed
                            cache_status_msg = "Installed packages changed since cache was created. Invalidating cache."
                            logger.info(cache_status_msg)
                            # Let cached_outdated_str remain None to trigger live fetch
                    else:
                        # Cache entry exists but has expired
                        expiry_dt = datetime.fromtimestamp(expiry_timestamp)
                        cache_status_msg = f"Cache found but expired at {expiry_dt.strftime('%Y-%m-%d %H:%M:%S')}. Fetching live data."
                        logger.info(cache_status_msg)
                else:
                    # Key not found in cache
                    cache_status_msg = "No cache found for outdated packages."
                    logger.info(cache_status_msg)
            except Exception as e:
                cache_status_msg = (
                    f"Failed to read from cache: {e}. Proceeding without cache."
                )
                logger.opt(exception=True).warning(cache_status_msg)
                if self.cache:
                    try:
                        self.cache.delete(cache_key)
                    except Exception as del_e:
                        logger.warning(
                            f"Failed to delete cache key '{cache_key}': {del_e}"
                        )

        elif not current_installed_hash:
            logger.warning(
                "Could not generate hash for current installed packages. Skipping cache validation based on package changes."
            )
        elif not self.cache or self.cache_duration_seconds <= 0:
            logger.info("Cache is disabled. Fetching live data.")

        # --- Fetch Data (if not cached or cache expired/invalid) ---
        if cached_outdated_str is None:
            self.stats.cache_used = False
            logger.info("Fetching live outdated package list from pip...")
            command = [
                "list",
                "--outdated",
                "--format=json",
                "--disable-pip-version-check",
            ]

            with timed_block("Fetch process"):
                live_outdated_str, stderr, return_code = self._run_pip_command(command)

            is_actual_error = (
                return_code != 0
                and "error" in stderr.lower()
                and "deprecationwarning" not in stderr.lower()
            )
            if is_actual_error:
                raise PipCommandError(
                    command=" ".join(command), stderr=stderr, return_code=return_code
                )
            elif return_code != 0:
                logger.warning(
                    "Pip list --outdated returned non-zero code (likely indicates updates found), but no specific error detected in stderr. Proceeding."
                )

            cached_outdated_str = live_outdated_str  # Use the live output

            # --- Update Cache ---
            if (
                self.cache is not None
                and self.cache_duration_seconds > 0
                and current_installed_hash
            ):
                try:
                    value_to_cache = (live_outdated_str, current_installed_hash)
                    self.cache.set(
                        cache_key,
                        value_to_cache,
                        expire=self.cache_duration_seconds,
                        tag=OUTDATED_CACHE_TAG,
                    )
                    logger.info(
                        f"Stored fresh outdated package list in cache (expires in {self.cache_duration_seconds}s)."
                    )
                except Exception as e:
                    logger.opt(exception=True).warning(f"Failed to write to cache: {e}")
            elif not current_installed_hash:
                logger.warning(
                    "Skipping cache update because current installed package hash could not be generated."
                )

        # --- Parse Data ---
        if not cached_outdated_str or not cached_outdated_str.strip():
            logger.success("No outdated packages found.")
            self.stats.outdated_count = 0
            return []

        try:
            outdated_data = json.loads(cached_outdated_str)
            packages = []
            for pkg in outdated_data:
                try:
                    pkg_name = pkg["name"]
                    current_v = pkg["version"]
                    latest_v = pkg["latest_version"]
                    parse_version(current_v)
                    parse_version(latest_v)
                except (InvalidVersion, TypeError):
                    logger.warning(
                        f"Skipping package '{pkg.get('name', 'Unknown')}' due to invalid version format (Current: '{current_v}', Latest: '{latest_v}')."
                    )
                    continue
                except KeyError as ke:
                    logger.warning(
                        f"Skipping package due to missing key {ke} in outdated data: {pkg}"
                    )
                    continue

                packages.append(
                    OutdatedPackageInfo(
                        name=pkg_name,
                        version=current_v,
                        latest_version=latest_v,
                        latest_filetype=pkg.get("latest_filetype", "unknown").lower(),
                        location=pkg.get("location", "unknown"),
                        installer=pkg.get("installer", "unknown"),
                    )
                )

            count = len(packages)
            source = "(from cache)" if self.stats.cache_used else "(live)"
            logger.success(f"Found {count} outdated packages {source}.")
            self.stats.outdated_count = count
            return packages
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output for outdated packages: {e}")
            logger.debug(f"Raw JSON data:\n{cached_outdated_str}")
            raise PipUpdaterError("Could not parse list of outdated packages.") from e

    def _filter_packages_to_update(self: Self) -> List[OutdatedPackageInfo]:
        """Filters outdated packages based on exclusion criteria."""
        logger.info(
            "Filtering outdated packages (excluding sdists, specified packages, and pip)..."
        )
        filtered_packages = []
        sdist_count = 0
        excluded_other_count = 0

        for pkg in self.outdated_packages:
            logger.debug(
                f"Checking package: {pkg.name} (Current: {pkg.version}, Latest: {pkg.latest_version}, Type: {pkg.latest_filetype})"
            )
            skip_reason = None
            if pkg.name in self.exclude_packages:
                skip_reason = "explicitly excluded by user"
                excluded_other_count += 1
            elif pkg.latest_filetype == "sdist":
                skip_reason = "latest available version is sdist"
                sdist_count += 1
            elif pkg.name == "pip":
                skip_reason = "'pip' package (recommend updating separately)"
                excluded_other_count += 1

            if skip_reason:
                logger.info(f"Skipping update for '{pkg.name}': {skip_reason}.")
            else:
                logger.debug(f"Package '{pkg.name}' marked for update.")
                filtered_packages.append(pkg)

        self.stats.sdist_count = sdist_count
        self.stats.excluded_count = excluded_other_count
        self.stats.packages_to_update = [pkg.name for pkg in filtered_packages]
        count = len(filtered_packages)

        logger.success(f"Identified {count} packages to update after filtering.")
        if sdist_count > 0 or excluded_other_count > 0:
            logger.info(
                f"Skipped {sdist_count} sdist packages and {excluded_other_count} other excluded/pip packages."
            )
        return filtered_packages

    def _calculate_installed_pkgs_hash(self: Self) -> str | None:
        """
        Computes a SHA256 hash of the currently installed pip packages list.

        Returns:
            A hex digest string if successful, otherwise None if an error occurs.
        """
        try:
            installed_cmd = ["list", "--format=json", "--disable-pip-version-check"]
            installed_json_str, _, retcode = self._run_pip_command(installed_cmd)
            if retcode != 0 or not installed_json_str.strip():
                logger.warning(
                    "Could not get installed package list JSON for hashing; cache validation might be skipped."
                )
                return None
            else:
                try:
                    current_installed_hash = hashlib.sha256(
                        installed_json_str.encode("utf-8", errors="strict")
                    ).hexdigest()
                    logger.debug(
                        f"Installed packages hash: '{current_installed_hash[:10]}...{current_installed_hash[-10:]}'"
                    )
                    return current_installed_hash
                except UnicodeEncodeError:
                    logger.error("Failed to encode JSON string with installed packages")
                    return None

        except Exception as hash_e:
            logger.warning(
                f"Failed to generate hash for installed packages: {hash_e}. Cache validation might be skipped."
            )
            return None

    def check_updates(self: Self) -> List[OutdatedPackageInfo]:
        """
        Checks for outdated packages, applying filtering and using cache.
        Cache validity now also depends on the installed package list remaining unchanged.
        """
        logger.info("Starting Package Update Check")
        cache_status = self.stats.cache_used
        self.stats = UpdateStats(start_time=datetime.now(), cache_used=cache_status)
        current_installed_hash = None

        try:
            self.installed_packages = self._get_installed_packages()
            # Generate a hash representing the current installed state for cache validation
            # Use the JSON output string for consistent hashing
            current_installed_hash = self._calculate_installed_pkgs_hash()
            self.outdated_packages = self._get_outdated_packages(current_installed_hash)
            self.packages_to_update = self._filter_packages_to_update()
        except (PipCommandError, PipUpdaterError, CacheError) as e:
            logger.error(f"Failed to check for updates: {e}")
            raise
        except Exception as e:
            logger.opt(exception=True).critical(
                f"An unexpected error occurred during update check: {e}"
            )
            raise PipUpdaterError("Unexpected error during update check.") from e

        logger.info("Package Update Check Finished")
        return self.packages_to_update

    def _parse_download_size(self: Self, output: str) -> int:
        """
        Parses pip's output (stdout/stderr combined) to find download sizes.
        """

        # Looks for: Downloading <any chars> ( <size> <unit> )
        size_pattern = re.compile(
            r"Downloading\s+.*\s+\(\s*([\d.]+)\s*([kKmMgG]B)\s*\)"
        )
        total_bytes = 0
        lines_checked = 0
        download_lines_found = 0

        output_lines = output.splitlines()
        lines_checked = len(output_lines)

        for line_num, line in enumerate(output_lines):
            match = size_pattern.search(line)
            if match:
                download_lines_found += 1
                size_str, unit_str = match.groups()
                try:
                    size = float(size_str)
                    unit = unit_str.upper()
                    multiplier = 1
                    if unit == "KB":
                        multiplier = 1024
                    elif unit == "MB":
                        multiplier = 1024 * 1024
                    elif unit == "GB":
                        multiplier = 1024 * 1024 * 1024
                    else:
                        logger.warning(
                            f"Unrecognized size unit '{unit_str}' in matched line: {line}"
                        )
                        continue  # Skip this match if unit is unknown

                    size_bytes = int(size * multiplier)
                    total_bytes += size_bytes
                except ValueError:
                    logger.warning(
                        f"Could not parse size value '{size_str}' as float in line: {line}"
                    )
                except Exception as e:
                    logger.opt(exception=True).warning(
                        f"Error processing matched download size line '{line}': {e}"
                    )

        # Log summary of parsing attempt
        if download_lines_found > 0:
            logger.debug(
                f"Parsed {download_lines_found} download line(s), total estimated size for this operation: {total_bytes / 1024:.2f} kB"
            )
        else:
            logger.debug(
                f"Checked {lines_checked} lines, but no 'Downloading...' lines matching the expected format were found. Pip might have used cached files."
            )

        return total_bytes

    def _check_for_conflicts(self: Self, stderr: str, package_name: str) -> bool:
        """
        Checks pip's stderr for dependency conflict errors SPECIFIC to the target package.

        Args:
             stderr: The standard error output from the pip command.
             package_name: The name of the package being installed/checked.

         Returns:
             True if a relevant conflict is detected, False otherwise.
        """
        if not stderr:
            return False

        # Normalize package name for comparison (handle underscores/hyphens)
        normalized_pkg_name = package_name.replace("_", "-").lower()
        # Regex to find common conflict error lines mentioning the specific package
        # Looks for lines containing error keywords AND the package name (case-insensitive, hyphen/underscore insensitive)
        # Example lines:
        # ERROR: Cannot install package-name because these package versions have conflicting dependencies.
        # ERROR: package-a 1.0 has requirement package-name<2.0,>=1.0, but you have package-name 2.1.
        # ERROR: ResolutionImpossible: ... involves package-name ...
        conflict_pattern = re.compile(
            # Match common error indicators at the start of the line or specific keywords
            r"(?:(?:ERROR: Cannot install)|(?:ERROR:.*?has requirement)|(?:ERROR: ResolutionImpossible)|(?:ERROR: Conflicting requirements)|(?:ERROR: pip's dependency resolver does not currently take into account.*?)).*?"
            # Ensure the package name (or normalized version) is mentioned in the error context
            # Use word boundaries (\b) to avoid partial matches (e.g., 'requests' matching 'requests-toolbelt')
            # Handle potential variations like pkg-name==version
            + r"\b"
            + re.escape(normalized_pkg_name).replace(r"\-", r"[-_]")
            + r"(?:[\s=!<>]|$)",
            re.IGNORECASE | re.MULTILINE,
        )

        # Normalize stderr
        # Replace newline/carriage return with space
        stderr = stderr.replace("\n", " ").replace("\r", " ")
        # Remove all characters except alphanumerics, spaces, and allowed specials
        allowed_specials: str = r"=._:\[\],<>\-@/+\\#'"
        allowed_pattern: str = rf"[^a-zA-Z0-9{re.escape(allowed_specials)} ]"
        stderr = re.sub(allowed_pattern, "", stderr)
        # Collapse multiple spaces and trim
        stderr = re.sub(r"\s+", " ", stderr).strip()

        match_ = conflict_pattern.search(stderr)
        if match_:
            # Log the specific line causing the match for debugging
            relevant_line = match_.group(0).splitlines()[
                0
            ]  # Get the first line of the match
            logger.warning(
                f"Detected potential conflict for '{package_name}' based on pattern in stderr line: \"{relevant_line}...\""
            )
            return True
        else:
            logger.debug(
                f"Package '{package_name}' not detected in stderr message, proceeding"
            )

        stderr_lower = stderr.lower()
        generic_conflict_patterns = [
            r"resolutionimpossible",
            r"cannot install .* conflicting dependencies",
            r"unable to resolve dependency tree",
            r"conflicting requirements",
            r"package .* has requirement .*, but you have",
            r"error: resolutionimpossible",
            r"error: conflicting requirements detected",
            r"failed to build dependency tree",
        ]
        for pattern in generic_conflict_patterns:
            if re.search(pattern, stderr_lower):
                logger.warning(
                    f"Detected potential dependency conflict pattern <{pattern}> in stderr."
                )
                return True
        return False

    def _check_for_externally_managed_env_error(self: Self, stderr: str) -> bool:
        """
        Detects if pip failed due to an 'externally managed environment' restriction.

        This typically occurs on modern Linux systems where pip is restricted from modifying
        system-managed Python environments unless '--break-system-packages' is explicitly used.

        Args:
            stderr: The stderr output from a pip command.

        Returns:
            True if the error is detected, False otherwise.
        """
        if not stderr:
            return False

        stderr_lower = stderr.lower()
        patterns = (
            r"externally[-_\s]managed[-_\s]environment",
            r"this environment is externally managed",
            r"pip cannot modify this environment",
            r"system-managed python environment",
        )

        if any(re.search(pat, stderr_lower) for pat in patterns):
            logger.error(
                "Detected 'externally managed environment' error. "
                "Use '--allow-break-system-packages' if you understand the risks."
            )
            return True

        return False

    def _attempt_cache_deletion(
        self, cache_key: str, package_name: Optional[str], reason: str
    ) -> None:
        """Helper to safely attempt cache deletion after an error."""
        log_pkg_info = f" for '{package_name}'" if package_name else ""
        try:
            logger.warning(
                f"Attempting to delete cache entry '{cache_key}' due to {reason}."
            )
            self.cache.delete(cache_key)
            logger.warning(f"Successfully deleted cache entry '{cache_key}'.")
        except Exception as inner_e:
            # Handle errors specific to cache deletion
            error_msg = (
                f"Failed to delete cache entry '{cache_key}' after {reason}: {inner_e}"
            )
            pruning_error = CachePruningError(package_name, error_msg)
            logger.error(str(pruning_error), level="error")

    def update_packages(self: Self) -> UpdateStats:
        """Attempts to update the filtered list of packages."""
        if not self.stats.start_time:
            logger.info("Statistics not initialized. Running check_updates first.")
            self.check_updates()

        if not self.packages_to_update:
            logger.warning("No packages identified for update. Nothing to do.")
            if not self.stats.end_time:
                self.stats.end_time = datetime.now()
            self._log_summary()
            return self.stats

        package_count = len(self.packages_to_update)
        logger.info(f"Starting Package Update Process for {package_count} packages")
        self.stats.attempted_update_count = package_count

        if self._allow_break_system_pkgs:
            logger.warning(
                "⚠️  '--break-system-packages' is enabled. "
                "This may irreversibly modify system-managed Python packages. "
                "Use only if you *absolutely* understand the implications."
                "See: https://pip.pypa.io/en/stable/topics/externally-managed-environments/"
            )

        if self.packages_to_update:
            self._analyzer.load_current_environment_state()
            logger.debug("Initialized environment state")

        progress_context: Any = contextlib.nullcontext()
        task_id = None
        if self.use_rich_console:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            )
            progress_context = progress
            task_id = progress.add_task(
                "[cyan]Updating packages...", total=package_count
            )

        # Retrieve current cache data if cache is enabled
        cache_key = self._get_cache_key()
        cached_outdated_data: Optional[List[Dict[str, Any]]] = None
        if self.cache and self.cache_duration_seconds > 0:
            logger.debug(f"Attempting to read cache for key: '{cache_key}'")
            try:
                cached_value = self.cache.get(
                    cache_key, default=None, expire_time=True, tag=OUTDATED_CACHE_TAG
                )
                if cached_value is not None:
                    (cached_data_str, _), expiry_timestamp, _ = cached_value
                    if isinstance(cached_data_str, str):
                        if not cached_data_str or not cached_data_str.strip():
                            # Cache entry exists but is empty string.
                            logger.debug(
                                f"Cache key '{cache_key}' is empty. No pruning needed."
                            )
                        else:
                            cached_outdated_data = json.loads(cached_data_str)
                            logger.debug(
                                "Successfully read outdated packages from cache."
                            )
                    else:
                        logger.warning(
                            "Cached data is not a string, skipping cache modification."
                        )
                        cached_outdated_data = None
            except json.JSONDecodeError as e:
                error_msg = (
                    f"JSON decode error during pruning parse for '{cache_key}': {e}"
                )
                logger.error(
                    f"{error_msg} — Cache entry is corrupt. Attempting deletion"
                )
                self._attempt_cache_deletion(
                    cache_key, None, "corrupt JSON during pruning parse"
                )
                cached_outdated_data = None
            except Exception as e:
                error_msg = (
                    f"Unexpected error during pruning processing for '{cache_key}': {e}"
                )
                logger.error(f"{error_msg} — Attempting deletion.")
                self._attempt_cache_deletion(
                    cache_key, None, "processing error during pruning"
                )
                cached_outdated_data = None

        with progress_context as progress:
            for i, pkg in enumerate(self.packages_to_update):
                package_name = pkg.name
                old_version = pkg.version
                new_version = pkg.latest_version
                update_failed = False
                failure_reason = "Unknown"
                is_conflict_failure = False

                if progress:
                    progress.update(
                        task_id,
                        description=f"[cyan]Checking {package_name} ({old_version} → {new_version})",
                    )

                # --- Dry Run Check ---
                logger.info(
                    f"Performing dry run check for {package_name}=={new_version}..."
                )
                dry_run_command = [
                    "install",
                    "--dry-run",
                    "--upgrade",
                    f"{package_name}=={new_version}",
                    "--disable-pip-version-check",
                    "--no-input",
                ]
                if self._allow_break_system_pkgs:
                    dry_run_command.append("--break-system-packages")

                dry_run_stdout, dry_run_stderr, dry_run_return_code = (
                    self._run_pip_command(dry_run_command)
                )

                # Check for conflicts specifically related to this package using dependency analyzer
                dry_run_conflict = self._analyzer.check_package_conflict(
                    package_name, new_version
                )

                if dry_run_conflict:
                    self.stats.conflict_detected_count += 1  # Count detected conflicts
                    if not self.ignore_conflicts:
                        logger.error(
                            f"Dependency conflict detected during dry run for '{package_name}'. Update prohibited."
                        )
                        update_failed = True
                        is_conflict_failure = True
                        failure_reason = ", ".join(dry_run_conflict)
                        self.stats.conflicted_packages.append(
                            package_name
                        )  # Add to list of packages skipped due to conflict
                        # Skip actual install attempt for this package
                    else:
                        logger.warning(
                            f"Dependency conflict detected during dry run for '{package_name}', but proceeding due to --ignore-conflicts flag."
                        )
                        self.stats.conflict_ignored_count += 1
                        # Proceed to actual install, but be aware it might still fail
                elif dry_run_return_code != 0:
                    # Distinguish from explicit conflicts - could be network error, package not found etc.
                    logger.warning(
                        f"Dry run for '{package_name}=={new_version}' failed (code: {dry_run_return_code}). This might indicate issues even without explicit conflicts."
                    )

                # Check for externally managed env errors
                ext_managed_error = self._check_for_externally_managed_env_error(
                    dry_run_stderr
                )

                if ext_managed_error and not self._allow_break_system_pkgs:
                    self.stats.system_managed_count += 1
                    logger.debug(
                        f"Package '{package_name}' is managed externally (update failed)"
                    )
                    update_failed = True
                    failure_reason = "System-managed package (dry-run)"
                    self.stats.system_packages.append(package_name)

                if not update_failed:
                    if progress:
                        progress.update(
                            task_id,
                            description=f"[cyan]Updating {package_name} ({old_version} → {new_version})",
                        )

                    logger.info(
                        f"Attempting update for: {package_name} ({old_version} -> {new_version})"
                    )
                    command = [
                        "install",
                        "--upgrade",
                        f"{package_name}=={new_version}",
                        "--disable-pip-version-check",
                        "--no-input",
                    ]
                    if self._allow_break_system_pkgs:
                        command.append("--break-system-packages")

                    stdout, stderr, return_code = self._run_pip_command(command)
                    # Post-install conflict check (as a safety net, though dry-run should catch most)
                    # Check only if install command succeeded (return_code == 0) but might have produced warnings
                    post_install_conflict = False
                    if return_code == 0:
                        post_install_conflict = self._check_for_conflicts(
                            stderr, package_name
                        )

                    if post_install_conflict:
                        # This case means install reported success (code 0) but stderr had conflict warnings
                        self.stats.conflict_detected_count += 1
                        if not self.ignore_conflicts:
                            logger.error(
                                f"Dependency conflict warning detected in stderr after successful install of '{package_name}'. Environment might be inconsistent. Update marked as FAILED."
                            )
                            logger.debug(
                                f"Post-install conflict details (stderr): {stderr}"
                            )
                            update_failed = True
                            is_conflict_failure = (
                                True  # Treat this as a conflict failure
                            )
                            failure_reason = "Dependency conflict warning post-install"
                            self.stats.conflicted_packages.append(
                                package_name
                            )  # Add to skipped list
                            # Note: Package IS installed at this point, unlike dry-run failure. Manual intervention likely needed.
                        else:
                            logger.warning(
                                f"Dependency conflict warning detected post-install for '{package_name}', but ignored due to flag."
                            )
                            self.stats.conflict_ignored_count += 1

                    if not update_failed:
                        if return_code == 0:
                            # Downloaded from dry-run, actual install uses pip's cached version
                            download_bytes = self._parse_download_size(
                                dry_run_stdout + dry_run_stderr
                            )
                            self.stats.total_download_size_bytes += download_bytes
                            if download_bytes > 0:
                                logger.debug(
                                    f"Estimated {download_bytes / 1024 ** 2:.7f} MB downloaded for {package_name} update."
                                )

                            try:
                                current_info = self._get_specific_package_info(
                                    package_name
                                )
                                if current_info and current_info.version == new_version:
                                    logger.success(
                                        f"Successfully updated {package_name} to {new_version}"
                                    )
                                    self.stats.successful_update_count += 1
                                    self.stats.updated_packages[package_name] = (
                                        old_version,
                                        new_version,
                                    )
                                    if cached_outdated_data is not None:
                                        logger.debug(
                                            f"Attempting to remove {package_name} from cache data."
                                        )
                                        # Remove the successfully updated package from the cached list
                                        initial_count = len(cached_outdated_data)
                                        cached_outdated_data = [
                                            item
                                            for item in cached_outdated_data
                                            if item.get("name") != package_name
                                        ]
                                        if len(cached_outdated_data) < initial_count:
                                            logger.success(
                                                f"Removed `{package_name}` from cache data."
                                            )
                                        else:
                                            logger.debug(
                                                f"`{package_name}` not found in cache data thus not removed"
                                            )

                                elif current_info:
                                    logger.error(
                                        f"pip reported success for {package_name}, but installed version is {current_info.version} (expected {new_version}). Marking as failed."
                                    )
                                    update_failed = True
                                    failure_reason = f"Version mismatch after update ({current_info.version} installed)"
                                else:
                                    logger.error(
                                        f"pip reported success for {package_name}, but could not verify installed version after update. Marking as failed."
                                    )
                                    update_failed = True
                                    failure_reason = (
                                        "Could not verify version after update"
                                    )
                            except PipUpdaterError as verify_err:
                                logger.error(
                                    f"pip reported success for {package_name}, but failed to verify installed version ({verify_err}). Marking as failed."
                                )
                                update_failed = True
                                failure_reason = f"Verification failed ({verify_err})"
                        else:
                            logger.error(
                                f"Failed to update {package_name}. Pip command failed with return code: {return_code}"
                            )
                            update_failed = True
                            if not is_conflict_failure and self._check_for_conflicts(
                                stderr, package_name
                            ):
                                failure_reason = f"Pip command failed (code {return_code}, likely due to conflict)"
                                is_conflict_failure = True  # Mark as conflict if pattern matches on failure
                                # Add to conflicted list only if not ignoring conflicts
                                if not self.ignore_conflicts:
                                    self.stats.conflicted_packages.append(package_name)
                            else:
                                failure_reason = (
                                    f"Pip command failed (code {return_code})"
                                )

                if update_failed:
                    self.stats.failed_update_count += 1
                    self.stats.failed_packages[package_name] = failure_reason

                if progress:
                    progress.update(task_id, advance=1)

        # Save the modified cached data back to the cache
        if (
            cached_outdated_data is not None
            and self.cache
            and self.cache_duration_seconds > 0
        ):
            logger.debug(
                f"Attempting to update cache for key (post update): '{cache_key}'"
            )
            try:
                # Re-serialize the modified list and update the cache with remaining TTL
                remaining_ttl: int = int(expiry_timestamp - time.time())
                updated_hash: str | None = self._calculate_installed_pkgs_hash()
                if not updated_hash:
                    raise ValueError("Empty hash.")

                if remaining_ttl <= 0:
                    logger.debug(
                        f"Cache key '{cache_key}' is already expired or about to expire. No pruning needed, entry will be refreshed on next check."
                    )
                else:
                    updated_value = (json.dumps(cached_outdated_data), updated_hash)
                    self.cache.set(
                        cache_key,
                        updated_value,
                        expire=remaining_ttl,
                        tag=OUTDATED_CACHE_TAG,
                    )
                    logger.info("Cache updated with remaining outdated packages")
            except TypeError as e:
                error_msg = f"Unexpected data structure during pruning parse for '{cache_key}': {e}"
                logger.error(
                    f"{error_msg} — Cache entry has unexpected structure. Attempting deletion."
                )
                self._attempt_cache_deletion(
                    cache_key, None, "malformed structure during pruning parse"
                )
            except ValueError as ve:
                error_msg = f"Failed to generate SHA256 hash for current list of installed packages for '{cache_key}': {ve}"
                logger.error(f"{error_msg} — Unable to update cache without hash.")
                self._attempt_cache_deletion(
                    cache_key, None, "processing error during hash computation"
                )
            except Exception as e:
                error_msg = f"Unexpected error during pruning serialization for '{cache_key}': {e}"
                logger.error(f"{error_msg} — Attempting deletion.")
                self._attempt_cache_deletion(
                    cache_key, None, "processing error during pruning"
                )

        self.stats.end_time = datetime.now()
        logger.info("Package Update Process Finished")
        self._log_summary()
        return self.stats

    def _get_specific_package_info(
        self: Self, package_name: str
    ) -> Optional[PackageInfo]:
        """Gets information for a single installed package using `pip show`."""
        logger.debug(f"Verifying installation status for package: {package_name}")
        command = ["show", package_name]
        stdout, stderr, return_code = self._run_pip_command(command)

        if return_code != 0:
            logger.warning(
                f"Could not get info for package '{package_name}' using 'pip show'. It might not be installed correctly. Stderr: {stderr}"
            )
            return None

        info = {}
        try:
            for line in stdout.splitlines():
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip().lower().replace("-", "_")
                    value = parts[1].strip()
                    info[key] = value

            if "name" in info and "version" in info:
                return PackageInfo(
                    name=info["name"],
                    version=info["version"],
                    location=info.get("location", "unknown"),
                    installer=info.get("installer", "unknown"),
                )
            else:
                logger.warning(
                    f"Could not parse required fields (name, version) from 'pip show {package_name}' output."
                )
                logger.debug(f"Parsed 'pip show' info: {info}")
                return None
        except Exception as e:
            logger.opt(exception=True).error(
                f"Error parsing 'pip show {package_name}' output: {e}"
            )
            logger.debug(f"Raw 'pip show' output:\n{stdout}")
            return None

    def _log_summary(self: Self) -> None:
        """Logs a summary report of the update process, using Rich tables if enabled."""
        logger.info("=" * 45)
        logger.info(f"{'Update Summary Report'.center(45)}")
        stats = self.stats
        duration = stats.duration
        duration_str = f"{duration:.2f} seconds" if duration is not None else "N/A"
        total_dl_mb = stats.total_download_size_mb
        logger.info("=" * 45)
        logger.info(f"Process duration: {duration_str}")
        logger.info(f"Outdated check cache used: {'Yes' if stats.cache_used else 'No'}")
        logger.info(f"Total packages checked: {stats.checked_count}")
        logger.info(f"Outdated packages found: {stats.outdated_count}")
        logger.info(f"Packages skipped (sdist): {stats.sdist_count}")
        logger.info(f"Packages skipped (excluded/pip): {stats.excluded_count}")
        logger.info(f"Packages attempted to update: {stats.attempted_update_count}")
        logger.info(f"Successfully updated: {stats.successful_update_count}")
        logger.info(f"Failed to update: {stats.failed_update_count}")
        logger.info(f"Dependency conflicts detected: {stats.conflict_detected_count}")
        if stats.conflict_detected_count > 0:
            logger.info(
                f"- Conflicts ignored (due to --ignore-conflicts): {stats.conflict_ignored_count}"
            )
            skipped_conflict_count = len(stats.conflicted_packages)
            logger.info(f"- Updates skipped due to conflicts: {skipped_conflict_count}")
        logger.info(
            f"Externally managed environment packages: {stats.system_managed_count}"
        )
        logger.info(f"Total estimated download size: {total_dl_mb:.2f} MB")

        if self.use_rich_console:
            if stats.updated_packages:
                success_table = Table(
                    title="[bold green]Successful Updates[/]",
                    show_header=True,
                    header_style="bold blue",
                    expand=True,
                )
                success_table.add_column(
                    "Package", style="cyan", width=30, no_wrap=True
                )
                success_table.add_column("Old Version", style="yellow")
                success_table.add_column("New Version", style="green")
                for name, (old_v, new_v) in sorted(stats.updated_packages.items()):
                    success_table.add_row(name, old_v, new_v)
                self.console.print(success_table)
            else:
                logger.info("No packages were successfully updated.")

            if stats.failed_packages:
                fail_table = Table(
                    title="[bold red]Failed Updates[/]",
                    show_header=True,
                    header_style="bold red",
                    expand=True,
                )
                fail_table.add_column("Package", style="cyan", width=30, no_wrap=True)
                fail_table.add_column("Reason", style="red")
                for name, reason in sorted(stats.failed_packages.items()):
                    if "conflict detected" in reason.lower():
                        fail_table.add_row(
                            f"[bold yellow]{name}[/]", f"[yellow]{reason}[/]"
                        )
                    elif "system-managed package" in reason.lower():
                        fail_table.add_row(
                            f"[bold red]{name}[/]", f"[orange]{reason}[/]"
                        )
                    else:
                        fail_table.add_row(f"[bold red]{name}[/]", reason)
                self.console.print(fail_table)

            if stats.conflicted_packages and not self.ignore_conflicts:
                skipped_conflict_str = ", ".join(
                    f"[yellow]{pkg}[/]" for pkg in sorted(stats.conflicted_packages)
                )
                self.console.print(
                    f"[bold orange_red1]Updates skipped due to conflicts:[/bold orange_red1] {skipped_conflict_str}"
                )
            if stats.system_packages and not self._allow_break_system_pkgs:
                failed_packages_colored = ", ".join(
                    f"[yellow]{pkg}[/]" for pkg in sorted(stats.system_packages)
                )
                self.console.print(
                    f"[bold orange_red1]Updates skipped due to environment restrictions:[/bold orange_red1] {failed_packages_colored}"
                )
        else:  # Standard log output
            if stats.updated_packages:
                logger.info("Successfully updated packages:")
                for name, (old_v, new_v) in sorted(stats.updated_packages.items()):
                    logger.info(f"  - {name}: {old_v} -> {new_v}")
            else:
                logger.info("No packages were successfully updated.")

            if stats.failed_packages:
                logger.warning("Failed packages:")
                for name, reason in sorted(stats.failed_packages.items()):
                    logger.warning(f"  - {name}: {reason}")

            if stats.conflicted_packages and not self.ignore_conflicts:
                logger.warning("Updates skipped due to detected conflicts:")
                for name in sorted(stats.conflicted_packages):
                    logger.warning(f"  - {name}")

            if stats.system_packages and not self._allow_break_system_pkgs:
                logger.warning("Updates skipped due to environment restrictions:")
                for name in sorted(stats.system_packages):
                    logger.warning(f"  - {name}")


# --- CLI Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Pip Package Updater",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=80
        ),
        epilog=f"""
Default cache directory: {DEFAULT_CACHE_DIR}
Default cache duration: {DEFAULT_CACHE_DURATION_MINUTES} minutes

Example Usage:
  # Check only, use cache (default), show debug logs
  python {sys.argv[0]} --log-level DEBUG

  # Update packages, ignore conflicts, skip confirmation, disable cache entirely
  python {sys.argv[0]} --update --ignore-conflicts -y --no-cache

  # Check only, force refresh (equivalent to --no-cache)
  python {sys.argv[0]} --cache-duration 0

  # Check only, specify a custom cache directory and duration
  python {sys.argv[0]} --cache-dir /path/to/my/cache --cache-duration 120

  # Force update system-managed packages
  # ⚠️  Use with extreme caution
  python {sys.argv[0]} --update --allow-break-system-packages

""",
    )

    exclusion_group = parser.add_argument_group("Exclusion Options")
    logging_cache_group = parser.add_argument_group("Logging & Cache Options")
    execution_group = parser.add_argument_group("Execution Options")

    exclusion_group.add_argument(
        "--exclude-file",
        type=Path,
        metavar="FILE_PATH",
        help="Path to a text file containing package names to exclude (one per line).",
    )
    exclusion_group.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        metavar="PKG",
        help="List of package names to exclude directly.",
    )

    logging_cache_group.add_argument(
        "--log-level",
        type=str.upper,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the minimum console logging level (default: INFO).",
    )
    logging_cache_group.add_argument(
        "--log-file-path",
        type=Path,
        default=Path("pip_updater.log"),
        metavar="PATH",
        help="Path to the log file (default: pip_updater.log).",
    )
    logging_cache_group.add_argument(
        "--no-log-file",
        action="store_false",
        dest="log_to_file",
        help="Disable logging to a file.",
    )
    logging_cache_group.add_argument(
        "--no-rich-console",
        action="store_false",
        dest="rich_console",
        help="Disable rich formatting (colors, tables, progress) in console output.",
    )
    logging_cache_group.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable caching entirely (equivalent to --cache-duration 0).",
    )
    logging_cache_group.add_argument(
        "--cache-duration",
        type=int,
        default=DEFAULT_CACHE_DURATION_MINUTES,
        metavar="MINUTES",
        help=f"Cache validity duration for outdated checks in minutes (0 disables cache, default: {DEFAULT_CACHE_DURATION_MINUTES}).",
    )
    logging_cache_group.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="PATH",
        help=f"Directory for cache files (default: {DEFAULT_CACHE_DIR}).",
    )

    execution_group.add_argument(
        "--update",
        action="store_true",
        help="Perform package updates after checking. Default is check-only.",
    )
    execution_group.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="skip_confirmation",
        help="Skip the confirmation prompt before updating packages.",
    )
    execution_group.add_argument(
        "--ignore-conflicts",
        action="store_true",
        help="Attempt updates even if pip detects dependency conflicts (use with caution).",
    )
    execution_group.add_argument(
        "--allow-break-system-packages",
        action="store_true",
        help=(
            "Allow pip to upgrade system-managed packages by using the "
            "`--break-system-packages` flag. "
        ),
    )

    args = parser.parse_args()

    # Check for conflicting cache arguments
    arg_strings = " ".join(sys.argv)
    cache_duration_specified = "--cache-duration" in arg_strings
    cache_dir_specified = "--cache-dir" in arg_strings

    if args.no_cache:

        if (
            cache_duration_specified and args.cache_duration != 0
        ):  # Allow specifying --cache-duration 0
            parser.error(
                "Cannot use --cache-duration (with non-zero value) when --no-cache is specified."
            )
        if cache_dir_specified:
            # Check if the specified dir is different from the default, as providing the default is harmless
            if args.cache_dir.resolve() != DEFAULT_CACHE_DIR.resolve():
                parser.error("Cannot use --cache-dir when --no-cache is specified.")
            else:
                print(
                    "Warning: --cache-dir specified with default value alongside --no-cache is redundant.",
                    file=sys.stderr,
                )

        # If --no-cache is set, enforce cache_duration = 0 internally
        args.cache_duration = 0
    elif args.cache_duration == 0:
        # If duration is 0, also log that cache is disabled for this run
        logger.debug("--cache-duration set to 0, caching disabled for this run.")

    log_file_arg_provided = any(arg.startswith("--log-file-path") for arg in sys.argv)
    if (
        not args.log_to_file
        and log_file_arg_provided
        and args.log_file_path != Path("pip_updater.log")
    ):
        print(
            f"Warning: --log-file-path '{args.log_file_path}' provided, but file logging is disabled via --no-log-file. The path will be ignored.",
            file=sys.stderr,
        )

    if args.exclude_file:
        try:
            resolved_exclude_file = args.exclude_file.resolve(strict=True)
            if not resolved_exclude_file.is_file():
                parser.error(f"Exclude path is not a file: {resolved_exclude_file}")
        except FileNotFoundError:
            parser.error(f"Exclude file not found: {args.exclude_file}")
        except Exception as e:
            parser.error(f"Error accessing exclude file '{args.exclude_file}': {e}")

    if args.cache_duration < 0:
        parser.error("--cache-duration cannot be negative.")

    return args


def read_exclude_file(file_path: Path) -> List[str]:
    """Reads package names from a file."""
    if not file_path:
        return []
    packages = set()
    print(f"Reading exclusion list from: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    if " " in line or "\t" in line:
                        print(
                            f"Warning: Possible invalid package name '{line}' (contains whitespace) in {file_path} at line {line_num}. Skipping.",
                            file=sys.stderr,
                        )
                        continue
                    packages.add(line)
        print(f"Read {len(packages)} unique package names from exclude file.")
        return list(packages)
    except IOError as e:
        print(f"Error: Could not read exclude file '{file_path}': {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(
            f"Error: An unexpected error occurred while reading exclude file '{file_path}': {e}",
            file=sys.stderr,
        )
        raise IOError(f"Unexpected error reading exclude file: {file_path}") from e


# --- Main Execution Block ---


def main():
    """Main function to parse arguments and run the updater."""
    args = parse_arguments()
    print(
        f"> Running PipUpdater CLI at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}..."
    )

    combined_exclusions = set(args.exclude)
    if args.exclude_file:
        try:
            file_exclusions = read_exclude_file(args.exclude_file)
            combined_exclusions.update(file_exclusions)
        except (IOError, argparse.ArgumentError) as e:
            print(f"Exiting due to error processing exclude file.", file=sys.stderr)
            sys.exit(1)

    updater: Optional[PipUpdater] = None
    try:
        updater = PipUpdater(
            exclude_packages=list(combined_exclusions),
            log_level=args.log_level,
            log_to_file=args.log_to_file,
            log_file_path=args.log_file_path,
            rich_console=args.rich_console,
            cache_duration_minutes=args.cache_duration,
            ignore_conflicts=args.ignore_conflicts,
            cache_dir=args.cache_dir,
            allow_break_system_pkgs=args.allow_break_system_packages,
        )
    except (CacheError, Exception) as e:
        print(f"\n--- FATAL ERROR during initialization ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)
        traceback.print_exc()
        print("-" * 40, file=sys.stderr)
        sys.exit(1)

    if not updater:
        print("Fatal Error: Updater object could not be created.", file=sys.stderr)
        sys.exit(1)

    exit_code = 0
    try:
        packages_needing_update = updater.check_updates()

        if not packages_needing_update:
            logger.info(
                "[bold green]All relevant packages are up-to-date.[/bold green]"
            )
            sys.exit(0)

        logger.info(
            f"[bold yellow]Found {len(packages_needing_update)} packages to update:[/bold yellow]"
        )
        for pkg in packages_needing_update:
            logger.info(
                f"• [cyan]{pkg.name}[/cyan] ({pkg.version} → [green]{pkg.latest_version}[/green])"
            )

        if args.update:
            logger.info("Update flag [--update] is set.")
            proceed = args.skip_confirmation
            if not proceed:
                try:
                    prompt_text = "\n[bold yellow]Proceed with updates? (y/N):[/] "
                    confirm = (
                        updater.console.input(prompt_text)
                        if updater.use_rich_console
                        else input("Proceed with updates? (y/N): ")
                    )
                    proceed = (
                        confirm.lower().strip() == "y" or confirm.lower().strip() == ""
                    )
                except EOFError:
                    logger.warning(
                        "Could not get confirmation (EOFError). Aborting update."
                    )
                    proceed = False
                except KeyboardInterrupt:
                    logger.warning("\nUpdate confirmation cancelled by user.")
                    proceed = False

            if proceed:
                logger.info("User confirmed. Proceeding with package updates...")
                final_stats = updater.update_packages()

                if final_stats.failed_update_count > 0 or (
                    len(final_stats.conflicted_packages) > 0
                    and not args.ignore_conflicts
                ):
                    logger.error(
                        "[bold red]Update process completed with failures or unignored conflicts.[/bold red]"
                    )
                    exit_code = 1
                else:
                    logger.success(
                        "[bold green]Update process completed successfully.[/bold green]"
                    )
                    exit_code = 0
            else:
                logger.info("Update process aborted by user confirmation.")
                exit_code = 0
        else:
            logger.info("Run with the --update flag to install these updates.")
            exit_code = 0

    except (PipUpdaterError, CacheError) as e:
        logger.critical(f"A critical error occurred: {e}")
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user (Ctrl+C).")
        exit_code = 1
    except Exception as e:
        logger.opt(exception=True).critical(f"An unexpected error occurred: {e}")
        exit_code = 1
    finally:
        logger.info(f"PipUpdater finished with exit code {exit_code}.")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
