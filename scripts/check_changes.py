"""Script to check that the CHANGES.md file was updated, and that the latest commit is newer than the target branch."""

import sys
import subprocess
from subprocess import PIPE
from datetime import datetime


def read_date(commit_message):
    """Read the date from a commit message."""

    for line in commit_message.split("\n"):
        if line.startswith("Date:"):
            date_str = line.split("Date:")[1].strip()

            date_format = "%a %b %d %H:%M:%S %Y %z"
            parsed_date = datetime.strptime(date_str, date_format)
            return parsed_date


# target branch
target_branch = sys.argv[1]  # first element in sys.argv is the script name

cmd = f"git fetch origin {target_branch}"
_ = subprocess.check_output(cmd, shell=True, text=True, stderr=PIPE)

# Get the last commit message of the current branch
cmd = "git log -n 1 CHANGES.md"
current_commit = subprocess.check_output(cmd, shell=True, text=True)
current_date = read_date(current_commit)

# Get the last commit message from the target branch
cmd = f"git log -n 1 origin/{target_branch} -- CHANGES.md"
target_commit = subprocess.check_output(cmd, shell=True, text=True)
target_date = read_date(target_commit)

# Check that the current commit is newer than the target commit
if current_date == target_date:
    print(f"The CHANGES.md file was not updated since the version on {target_branch}. Please update the CHANGES.md file.")
    sys.exit(1)  # exit with error
if current_date < target_date:
    print(f"The CHANGES.md file is not up-to-date. The version on {target_branch} was updated '{target_date}' but the current version was last updated '{current_date}'. Please update the CHANGES.md file.")
    sys.exit(1)  # exit with error
else:
    print(f"CHANGES.md file was updated (version on {target_branch} committed '{target_date}'; current version was committed '{current_date}').")
