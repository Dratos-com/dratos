import git
import os
from datetime import datetime

class GitAPI:
    def __init__(self, repo_path):
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
            self.repo = git.Repo.init(repo_path)
        else:
            self.repo = git.Repo(repo_path)

    def commit_memory(self, message=None):
        self.repo.git.add(A=True)
        commit_message = message or f"Memory snapshot at {datetime.utcnow().isoformat()}"
        commit = self.repo.index.commit(commit_message)
        return commit.hexsha

    def create_branch(self, branch_name):
        self.repo.git.branch(branch_name)

    def switch_branch(self, branch_name):
        self.repo.git.checkout(branch_name)

    def checkout_commit(self, commit_hash):
        self.repo.git.checkout(commit_hash)

    def get_current_branch(self):
        return self.repo.active_branch.name

    def list_branches(self):
        return [branch.name for branch in self.repo.branches]

    def list_commits(self, branch_name, max_count=10):
        commits = list(self.repo.iter_commits(branch_name, max_count=max_count))
        return [{'hash': commit.hexsha, 'message': commit.message} for commit in commits]