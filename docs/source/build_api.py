""" Script to generate the API for the sctoolbox package."""

import os
import glob


##############################################################################
# ---------------------------- Helper functions ---------------------------- #
##############################################################################

def hline():
    line = "-" * 30 + "\n\n"
    return line


def header(text, level):

    if level == 0:  # doc title
        s = "=" * len(text) + "\n"
        s += f"{text}\n"
        s += "=" * len(text) + "\n\n"

    else:
        level_symbol = {1: "=", 2: "-", 3: "~"}

        s = f"{text}\n"
        s += f"{level_symbol[level] * len(text)}\n\n"

    return s


def automodule(name):

    s = f"""
.. automodule:: {name}
    :members:
    :undoc-members:
    :show-inheritance:
    """
    s += "\n"

    return s


def get_modules(path):

    modules = [os.path.basename(f).replace(".py", "") for f in glob.glob(path + "/*")]
    modules = [f for f in modules if not f.startswith("_") and f != "data"]

    return modules


##############################################################################
# ---------------------------- Configure API ------------------------------- #
##############################################################################

package_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../sctoolbox/"

# ------- Setup categories within modules ------- #

# Choose category for each module
submodule_categories = {"tools": {"Preprocessing": ["qc_filter", "highly_variable", "dim_reduction", "norm_correct"],
                                  "Analysis": ["embedding", "clustering", "marker_genes", "celltype_annotation"],
                                  "Additional": []}}
assigned = sum(submodule_categories["tools"].values(), [])

# Add remainder of submodules to "Additional"
tool_submodules = get_modules(package_dir + "tools")
for submodule in tool_submodules:
    if submodule not in assigned:
        submodule_categories["tools"]["Additional"].append(submodule)

# ------- Setup order of submodules within modules ------- #
submodule_order_dict = {"plotting": ["qc_filter", "highly_variable", "embedding", "clustering", "marker_genes"],
                        "tools": ["qc_filter", "highly_variable", "dim_reduction", "norm_correct", "embedding", "clustering"]
                        }

# ---------- Set header of each module page ---------- #

page_headers = {}

page_headers["plotting"] = ".. rubric:: Loading example data\n\n"
page_headers["plotting"] += ".. plot ::\n\n"
page_headers["plotting"] += "    " + open("plot_pre_code.py").read().replace("\n", "\n    ") + "\n\n"
page_headers["plotting"] += hline() + "\n"


##############################################################################
# ------------- Main function for generating API documentation ------------- #
##############################################################################

def main():

    # Generate the API documentation per module
    modules = get_modules(package_dir)

    # Create one page per module
    for module in modules:
        with open("API/" + module + ".rst", 'w') as fp:

            fp.write(header(module.capitalize(), 1))

            # Write page header if it exists
            fp.write(page_headers.get(module, ""))

            # Check if module has categories
            submodule_categories_module = submodule_categories.get(module, None)
            if submodule_categories_module is None:

                # Find all submodules
                submodules = get_modules(package_dir + module)

                # reorder submodules based on preferred order of submodules (all additional submodules are added at the end)
                submodule_order = submodule_order_dict.get(module, [])
                submodules_ordered = [submodule for submodule in submodule_order if submodule in submodules]
                submodules_ordered += [submodule for submodule in submodules if submodule not in submodule_order]

                # Add submodules to rst file
                for submodule in submodules_ordered:
                    fp.write(header(submodule, 2))
                    fp.write(automodule(f"sctoolbox.{module}.{submodule}"))

                    if submodule != submodules[-1]:
                        fp.write(hline())  # add horizontal line between submodules

            else:
                for category, submodules in submodule_categories_module.items():
                    fp.write(header(category, 2))

                    for submodule in submodules:
                        fp.write(header(submodule, 3))
                        fp.write(automodule(f"sctoolbox.{module}.{submodule}"))

                        if submodule != submodules[-1]:
                            fp.write(hline())  # add horizontal line between submodules


if __name__ == "__main__":
    main()
