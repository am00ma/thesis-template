import subprocess
from pathlib import Path


def check_git_status():
    """
    Check the Git status of the current repository.

    Returns:
        str:
            - The commit hash (if clean).
            - The commit hash prefixed with 'dirty-' (if unstaged changes exist).

    Raises:
        FileNotFoundError: If git is not installed.
        subprocess.CalledProcessError: If git commands fail.
        RuntimeError: If not in a Git repository.
    """
    try:
        # Check if the current directory is a git repository
        if not Path(".git").exists():
            raise RuntimeError("Not a git repository")

        # Get the current commit hash (regardless of dirty state)
        commit_hash_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = commit_hash_result.stdout.strip()

        # Check if there are unstaged changes
        diff_result = subprocess.run(["git", "diff", "--quiet"], capture_output=True, text=True)

        # Return dirty-{hash} if unstaged changes exist (git diff returns 1)
        if diff_result.returncode == 1:
            return f"dirty-{commit_hash}"
        else:
            return commit_hash

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr.strip()}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError("Git is not installed or not in PATH") from e
