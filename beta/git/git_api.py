import subprocess
import os

class GitAPI:
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path

    def _run_git_command(self, command: List[str]) -> str:
        return subprocess.check_output(["git"] + command, cwd=self.repo_path).decode().strip()

    def commit_memory(self, message: str) -> str:
        self._run_git_command(["add", "lancedb_data/"])
        return self._run_git_command(["commit", "-m", message])

    def create_branch(self, branch_name: str):
        self._run_git_command(["checkout", "-b", branch_name])

    def switch_branch(self, branch_name: str):
        self._run_git_command(["checkout", branch_name])

    def checkout_commit(self, commit_hash: str):
        self._run_git_command(["checkout", commit_hash])