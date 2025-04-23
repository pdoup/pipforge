# pipforge
##  Pip Package Updater 🚀

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Stability: Experimental/Pragmatic](https://img.shields.io/badge/stability-experimental-orange.svg)](#1#%EF%B8%8F-disclaimer-read-this-the-hacky-nature)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **pragmatic**, single-file command-line tool to check for and update outdated Python packages installed via `pip`. It offers features like caching, best-effort conflict detection, and rich console output, designed for straightforward use.

## ✨ Overview

Tired of manually running `pip list --outdated` and then `pip install -U <package>`? This script automates the process with a focus on practical utility over architectural purity. It intelligently checks for outdated packages, provides clear summaries, and can perform updates while attempting to handle potential dependency conflicts.

**Key Goals:**

* ✅ **Check:** Identify outdated packages in your Python environment.
* 📈 **Update:** Upgrade packages to their latest versions.
* 🧠 **Best-Effort Intelligence:** Detect potential dependency conflicts using available tools (`pipdeptree`, dry runs) and output parsing.
* 💾 **Cache:** Speed up checks by caching outdated package lists.
* ✨ **Rich Output:** Present information clearly using tables and progress bars (optional).
* ⚙️ **Flexible:** Offer various command-line options for customization.

## ⚠️ Disclaimer: Read This! (The "Hacky" Nature)

This script is intentionally built using a **pragmatic and somewhat "hacky" approach**. Here's what that means:

* **Relies on CLI Output:** It heavily depends on parsing the text and JSON output of standard command-line tools like `pip` and `pipdeptree`.
* **Potential Brittleness:** If future versions of `pip` or `pipdeptree` significantly change their output format or error messages, **this script could break** or behave unexpectedly.
* **Regex-Based Detection:** Some checks (like conflict detection in stderr) rely on regular expressions matching specific error patterns, which might not catch every edge case or could have false positives.

**Why this approach?** It avoids adding complex dependencies on pip's internal APIs (which are not guaranteed to be stable) and keeps the tool self-contained. However, users should be aware of these limitations. **Use it with the understanding that it's a best-effort utility, not an infallible package management solution.** Always review planned changes, especially in critical environments.

## 🏛️ Single-File Architecture Rationale

You might notice this entire tool is contained within a single Python file. While not typical for larger projects, this was a conscious decision for a utility of this scope:

* **Portability:** Makes it incredibly easy to distribute and use. Just clone, and install it directly in your environment with `pip install`. No complex installation needed.
* **Simplicity for a Utility:** For a focused command-line tool, keeping everything together simplifies deployment and usage for end-users who just want a quick way to manage updates.
* **Self-Contained Logic:** The core logic is tightly coupled with interacting with `pip` commands, making separation less beneficial for this specific task compared to the ease of distribution.

## 📋 Features

* **Check & Update:** Lists outdated packages and optionally updates them.
* **Dependency Conflict Detection:** Uses `pipdeptree` and dry runs to identify potential version conflicts *before* attempting updates. *(Relies on tool output parsing)*.
* **Conflict Handling:** Choose to skip conflicting updates or attempt them anyway (`--ignore-conflicts`).
* **Caching:** Caches the results of `pip list --outdated` to speed up subsequent checks. Cache validity is based on duration and changes in installed packages.
* **Exclusions:** Exclude specific packages from checks and updates via command-line or an exclusion file.
* **Rich Console Output:** Uses the `rich` library for formatted tables, progress bars, and colored output. Can be disabled for simpler logs.
* **System Package Handling:** Option to allow updates in PEP 668 externally managed environments (`--allow-break-system-packages`).
* **Detailed Summary:** Provides statistics on checked, outdated, updated, failed, and skipped packages.

## 🛠️ Requirements

* **Python:** Version 3.11 or later is required.
* **Pip:** Needs `pip` and `pipdeptree` available in the environment.
* **Libraries:** The script uses external libraries. Install them using pip:

    ```bash
    pip install rich packaging diskcache loguru pipdeptree
    ```

## ⚙️ Usage

1.  **Show Help:**
    ```bash
    pipforge -h
    ```
    *(Output)*
    ```bash
    Pip Package Updater
    
    options:
      -h, --help                                             show this help message and exit
    
    Exclusion Options:
      --exclude-file FILE_PATH                               Path to a text file containing package names to exclude (one per line).
      --exclude [PKG ...]                                    List of package names to exclude directly.
    
    Logging & Cache Options:
      --log-level {TRACE,DEBUG,INFO,WARNING,ERROR,CRITICAL}  Set the minimum console logging level (default: INFO).
      --log-file-path PATH                                   Path to the log file (default: pip_updater.log).
      --no-log-file                                          Disable logging to a file.
      --no-rich-console                                      Disable rich formatting (colors, tables, progress) in console output.
      --no-cache                                             Disable caching entirely (equivalent to --cache-duration 0).
      --cache-duration MINUTES                               Cache validity duration for outdated checks in minutes (0 disables cache, default: 60).
      --cache-dir PATH                                       Directory for cache files (default: /home/user/.cache/pip_updater).
    
    Execution Options:
      --update                                               Perform package updates after checking. Default is check-only.
      -y, --yes                                              Skip the confirmation prompt before updating packages.
      --ignore-conflicts                                     Attempt updates even if pip detects dependency conflicts (use with caution).
      --allow-break-system-packages                          Allow pip to upgrade system-managed packages by using the `--break-system-packages` flag. 
    
    Default cache directory: /home/user/.cache/pip_updater
    Default cache duration: 60 minutes
    
    Example Usage:
      # Check only, use cache (default), show debug logs
      pipforge --log-level DEBUG
    
      # Update packages, ignore conflicts, skip confirmation, disable cache entirely
      pipforge --update --ignore-conflicts -y --no-cache
    
      # Check only, force refresh (equivalent to --no-cache)
      pipforge --cache-duration 0
    
      # Check only, specify a custom cache directory and duration
      pipforge --cache-dir /path/to/my/cache --cache-duration 120
    
      # Force update system-managed packages
      # ⚠️  Use with extreme caution
      pipforge --update --allow-break-system-packages
    ```
3.  **Check for Outdated Packages (Default Action):**
    ```bash
    pipforge
    ```
    *(This will list outdated packages but won't update anything)*

4.  **Check and Update Packages:**
    ```bash
    pipforge --update
    ```
    *(You will be prompted for confirmation before updates start)*

5.  **Update Without Confirmation:**
    ```bash
    pipforge --update -y
    # or
    pipforge --update --yes
    ```

6.  **Exclude Specific Packages:**
    ```bash
    # Exclude 'requests' and 'numpy' via command line
    pipforge --update --exclude requests numpy

    # Exclude packages listed in a file (one package name per line, '#' for comments)
    pipforge --update --exclude-file /path/to/exclude.txt
    ```

7.  **Ignore Conflicts (Use with Caution!):**
    ```bash
    pipforge --update --ignore-conflicts -y
    ```
    *(This attempts updates even if dependency conflicts are detected - see Disclaimer!)*

8.  **Allow Breaking System Packages (Use with Caution!):**
    ```bash
    # Needed on systems like Debian 12+ / Ubuntu 23.04+ if not using a venv
    pipforge --update --allow-break-system-packages -y
    ```

9.  **Disable Caching:**
    ```bash
    pipforge --no-cache
    # or
    pipforge --cache-duration 0
    ```

10.  **Customize Cache:**
    ```bash
    # Set cache duration to 2 hours (120 minutes)
    pipforge --cache-duration 120

    # Set a custom cache directory
    pipforge --cache-dir /tmp/my_pip_cache
    ```

11. **Disable Rich Formatting:**
    ```bash
    pipforge --no-rich-console
    ```

## 💾 Caching Explained

* **Purpose:** To avoid running the potentially slow `pip list --outdated` command every time.
* **How it Works:** The script stores the JSON output of `pip list --outdated` in a cache file. It also stores a hash of your *currently installed* packages list (`pip list --format=json`).
* **Validation:** The cache is considered valid only if:
    1.  The cache file hasn't expired (default: 60 minutes, configurable with `--cache-duration`).
    2.  The hash of your current environment's installed packages matches the hash stored when the cache was created. (This ensures the cache reflects the correct base state).
* **Location:** By default, cache files are stored in `~/.cache/pip_updater`. Use `--cache-dir` to change this.
* **Pruning:** When a package is successfully updated, the script *attempts* to remove that specific package from the cached outdated list. This makes subsequent checks (within the cache duration) faster if only some packages were updated.
* **Disabling:** Use `--no-cache` or `--cache-duration 0`.

## 🤝 Conflict Handling

Dependency conflicts can occur when updating a package requires a version of another package that clashes with the requirements of a *third* package.

1.  **Detection:**
    * The script runs `pipdeptree` initially to build a dependency graph.
    * Before updating package `X` to version `N`, it checks if any *other* installed package requires a version of `X` incompatible with `N`.
    * It also runs `pip install --dry-run` to let pip perform its own resolution check.
    * It analyzes pip's output (stderr) for common conflict error messages after both dry-run and actual install attempts (using regex matching, which is a best-effort approach).
2.  **Default Behavior:** If a conflict is detected for a package, the script will **skip** updating that package and report it in the summary.
3.  **Ignoring Conflicts (`--ignore-conflicts`):** If this flag is set, the script will still attempt the update even if conflicts are detected. Pip might still fail the installation, or you might end up with an inconsistent environment. **Use this option with caution and awareness of the potential risks!**

## 🚫 System Packages & PEP 668

Some Linux distributions (like recent Debian, Ubuntu, Fedora) mark their system Python environment as "externally managed" (PEP 668). By default, `pip` will refuse to install or update packages in these environments to prevent breaking system tools.

* **Detection:** The script checks pip's output for errors like "externally managed environment".
* **Default Behavior:** If this error is detected, the update for that package is skipped, and it's noted in the summary.
* **Overriding (`--allow-break-system-packages`):** This flag adds `--break-system-packages` to the `pip install` command. This tells pip to ignore the PEP 668 protection.

⚠️ **Warning:** Using this flag can potentially break system tools that rely on specific package versions. **It is strongly recommended to use virtual environments (`venv`, `conda`) instead of modifying the system Python directly.** Use this flag only if you understand the risks.

## 📄 Excluding Packages

You can prevent the script from checking or updating specific packages using two methods:

1.  **Command Line (`--exclude`):** Pass package names directly.
    ```bash
    pipforge --update --exclude Jinja2 Flask Werkzeug
    ```
2.  **Exclusion File (`--exclude-file`):** Provide a path to a text file containing one package name per line. Lines starting with `#` are ignored as comments.

    *Example `exclude.txt`:*
    ```txt
    # Core web framework - don't update automatically
    Flask
    Jinja2

    # Data science - manage manually
    numpy
    pandas

    # System specific - leave alone
    some-linux-tool
    ```
    *Usage:*
    ```bash
    pipforge --update --exclude-file ./exclude.txt
    ```

Packages listed in either way will be ignored during the outdated check and update process. The script also implicitly excludes `pip` itself and packages only available as source distributions (`sdist`) from automatic updates.

## ❌ Error Handling

The script includes error handling for common issues like:

* Missing dependencies (`rich`, `packaging`, `diskcache`, `loguru`).
* Incorrect Python version.
* Pip command failures.
* Cache directory access problems.
* Invalid command-line arguments.
* Errors reading exclusion files.

If critical errors occur, the script will attempt to print diagnostic information and exit with a non-zero status code. Given its reliance on external command output, unexpected errors are possible if `pip`'s behavior changes.

## 🔒 Security Considerations 

This script executes `pip install` commands. Therefore, it inherits the security implications of running `pip` and installing packages from your configured Python Package Index (usually PyPI). Ensure you trust the packages you are installing and the index you are using. This script does not introduce additional security vulnerabilities beyond those inherent in using `pip` itself.

## 🤝 Contributing

Given the pragmatic nature of this script, contributions focusing on robustness (e.g., improved error handling, better output parsing logic) are welcome. Feel free to open issues or submit pull requests!

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
