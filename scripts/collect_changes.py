# collect all changes associated with the given version

# Args:
# 1: path to changelog file (CHANGES.rst)
# 2: version to search for (e.g. 0.1.2)

import sys

changelog_path = sys.argv[1]
version = sys.argv[2]

with open(changelog_path, "r") as f:
    lines = f.readlines()

# The start and end line containing the changes of the version
start = -1
end = len(lines) - 1  # select the rest of the file if no end is found

# search for the start
for i, line in enumerate(lines):
    if line.startswith(version):
        start = i
        break
if start == -1:
    raise ValueError(f"Version {version} not found in changelog. Are the versions in sctoolbox/_version.py and CHANGES.rst matching?")

# search for the end
for i in range(start + 2, len(lines)):
    # look for the ====== marking the start of the next section
    if lines[i].startswith("="):
        # count empty lines above === line
        empty_count = 0
        for j in range(i - 2, start, -1):
            if lines[j].strip() == "":
                empty_count += 1
            else:
                break

        end = i - (1 + empty_count)
        break

# print the changes
for i in range(start, end):
    print(lines[i], end="")  # avoid double newlines as lines already contain \n
