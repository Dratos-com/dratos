import os
from typing import Optional, Tuple
from git import Repo, GitCommandError, BadName
from ulid import ULID
import asyncio


class GitAPI:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
        if not os.path.exists(os.path.join(repo_path, ".git")):
            self.repo = Repo.init(repo_path)
            # Create an initial commit
            open(os.path.join(repo_path, "README.md"), "w").close()
            self.repo.index.add(["README.md"])
            self.repo.index.commit("Initial commit")
        else:
            self.repo = Repo(repo_path)

    def create_branch(self, branch_name: str, commit_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        try:
            print(f"Creating branch: {branch_name} with commit_id: {commit_id}")
            
            if not self.repo.heads:
                # If there are no branches, create the first commit
                open(os.path.join(self.repo_path, "README.md"), "w").close()
                self.repo.index.add(["README.md"])
                commit = self.repo.index.commit("Initial commit")
                commit_id = commit.hexsha

            if commit_id:
                try:
                    commit = self.repo.commit(commit_id)
                    new_branch = self.repo.create_head(branch_name, commit)
                except BadName:
                    print(f"Invalid commit ID: {commit_id}. Creating branch at HEAD.")
                    new_branch = self.repo.create_head(branch_name)
            else:
                print(f"Creating branch: {branch_name} at HEAD")
                new_branch = self.repo.create_head(branch_name)
            
            new_branch.checkout()
            return branch_name, commit_id
        except (GitCommandError, BadName, ValueError) as e:
            print(f"Error creating branch: {e}")
            return None, None
        except TypeError as e:
            if "MemoryView" in str(e):
                print(
                    f"Error creating branch: MemoryView initialization failed. This might be due to an empty repository or an invalid commit ID."
                )
                return None, None
            raise

    async def create_branch_async(self, branch_name: str, commit_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        return await asyncio.to_thread(self.create_branch, branch_name, commit_id)

    def switch_branch(self, branch_name):
        if branch_name not in self.repo.heads:
            return self.create_branch(branch_name)
        return self.repo.heads[branch_name].checkout()

    def commit_memory(self, message: str) -> str:
        # Assuming we're writing memories to a file named 'memories.txt'
        with open(os.path.join(self.repo_path, "memories.txt"), "a") as f:
            f.write(message + "\n")
        self.repo.index.add(["memories.txt"])
        commit = self.repo.index.commit(f"Memory: {message}")
        return commit.hexsha

    def merge_branches(self, source_branch, target_branch):
        self.repo.git.checkout(target_branch)
        self.repo.git.merge(source_branch)

    def checkout_commit(self, commit_hash):
        self.repo.git.checkout(commit_hash)

    def get_branches(self):
        return self.repo.heads
