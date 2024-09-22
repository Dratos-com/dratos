import os
from git import Repo, GitCommandError, BadName

class GitAPI:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
        if not os.path.exists(os.path.join(repo_path, '.git')):
            self.repo = Repo.init(repo_path)
            # Create an initial commit
            open(os.path.join(repo_path, 'README.md'), 'w').close()
            self.repo.index.add(['README.md'])
            self.repo.index.commit("Initial commit")
        else:
            self.repo = Repo(repo_path)

    def create_branch(self, branch_name, commit_id=None):
        try:
            if not self.repo.heads:
                # If there are no branches, create the first commit
                open(os.path.join(self.repo_path, 'README.md'), 'w').close()
                self.repo.index.add(['README.md'])
                self.repo.index.commit("Initial commit")
            
            if branch_name not in self.repo.heads:
                if commit_id:
                    new_branch = self.repo.create_head(branch_name, commit_id)
                else:
                    new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
            return self.repo.heads[branch_name].checkout()
        except (GitCommandError, BadName) as e:
            print(f"Error creating branch: {e}")
            return None

    def switch_branch(self, branch_name):
        if branch_name not in self.repo.heads:
            return self.create_branch(branch_name)
        return self.repo.heads[branch_name].checkout()

    def commit_memory(self, message):
        # Assuming we're writing memories to a file named 'memories.txt'
        with open(os.path.join(self.repo_path, 'memories.txt'), 'a') as f:
            f.write(message + '\n')
        self.repo.index.add(['memories.txt'])
        commit = self.repo.index.commit(f"Memory: {message}")
        return commit.hexsha

    def merge_branches(self, source_branch, target_branch):
        self.repo.git.checkout(target_branch)
        self.repo.git.merge(source_branch)

    def checkout_commit(self, commit_hash):
        self.repo.git.checkout(commit_hash)

    def get_branches(self):
        return [head.name for head in self.repo.heads]