import git
import os
from datetime import datetime

class GitAPI:
    def __init__(self, repo_path):
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
            self.repo = git.Repo.init(repo_path)
        else:
            try:
                self.repo = git.Repo(repo_path)
            except git.exc.InvalidGitRepositoryError:
                self.repo = git.Repo.init(repo_path)

    def commit_memory(self, message):
        self.repo.git.add(A=True)
        commit = self.repo.index.commit(message)
        return commit.hexsha

    def create_branch(self, branch_name, commit_id='HEAD'):
        new_branch = self.repo.create_head(branch_name, commit_id)
        return new_branch

    def switch_branch(self, branch_name):
        self.repo.git.checkout(branch_name)

    def get_latest_commit_id(self):
        return self.repo.head.commit.hexsha

    def merge_branches(self, source_branch, target_branch):
        self.repo.git.checkout(target_branch)
        self.repo.git.merge(source_branch)

    def get_branches(self):
        return list(self.repo.heads)
