import os
import shutil
import subprocess
from pathlib import Path

import dotenv
from github import Github
from tqdm import tqdm

dotenv.load_dotenv()

WHITE_LIST = [
    "justinsasek",
    "github-classroom[bot]",
]  # Ignore commits by these authors when filtering repos
CLASSROOM_ORG = "MLDS-UT-AUSTIN"  # Replace with your org name
ASSIGNMENT_PREFIX = "longcompfall2025-"  # Replace with your assignment prefix
OUTPUT_DIR = "submissions"  # Directory to store cloned repos

# Please put your GITHUB_TOKEN in .env file
# read:org permission is required


def main():
    # Initialize GitHub API
    if not os.getenv("GITHUB_TOKEN"):
        raise ValueError("Please set GITHUB_TOKEN environment variable")
    g = Github(os.getenv("GITHUB_TOKEN"))
    org = g.get_organization(CLASSROOM_ORG)

    # clear the output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Files in '{OUTPUT_DIR}' will be deleted. Continue? [y/n]")
        if input().lower() != "y":
            return
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(parents=True)

    # Get all repos in the organization
    repos = org.get_repos()

    filtered_repos = []
    for repo in tqdm(list(repos), desc="Filtering repositories", colour="green"):
        # Check if repo name matches assignment prefix
        if not repo.name.startswith(ASSIGNMENT_PREFIX):
            continue

        # Get all commits
        commits = repo.get_commits()

        # Check if there are commits by other authors
        for commit in commits:
            author = commit.author.login if commit.author else None
            if author is None or author.lower() not in WHITE_LIST:
                filtered_repos.append(repo)
                break

    print(f"Cloning {len(filtered_repos)} out of {repos.totalCount} repositories")

    for repo in tqdm(filtered_repos, desc="Cloning repositories", colour="green"):
        # Clone into a subdirectory named after the repo
        repo_dir = os.path.join(OUTPUT_DIR, repo.name)

        try:
            subprocess.run(
                ["git", "clone", repo.clone_url, repo_dir],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning {repo.name}: {e}")


if __name__ == "__main__":
    main()
