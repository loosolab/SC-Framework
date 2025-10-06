import sys
import os
import glob
import papermill as pm

# get argument
# either RNA or ATAC
notebook_type = sys.argv[1]

# Get location of script
script_dir = os.path.dirname(__file__)
print(f"Script location: {script_dir}")

# notebook sorting key
key = lambda x: "zzzz" if os.path.basename(x).startswith("99") else str.lower(os.path.basename(x))  # put the 99-report.ipynb notebook last

if notebook_type == "RNA":
    # Run RNA notebooks
    rna_notebook_path_suffix = "/../rna_analysis/notebooks/"
    notebook_dir = script_dir + rna_notebook_path_suffix
    rna_notebooks = sorted(glob.glob(notebook_dir + "*.ipynb"), key=key)  # sort as glob output is not ordered
    # TODO remove the pseudotime notebook due to version conflict with scFates
    rna_notebooks = [n for n in rna_notebooks if os.path.basename(n) != "pseudotime_analysis.ipynb"]
    print("\n\n")
    print("--------------------------------------- RNA ---------------------------------------")
    print(f"Notebook directory: {notebook_dir}")
    print(f"Notebooks: {[os.path.basename(f) for f in rna_notebooks]}")
    print("-----------------------------------------------------------------------------------")
    print("\n\n")

    for notebook in rna_notebooks:
        # get filename
        notebook_file = os.path.basename(notebook)
        notebook_name = os.path.splitext(notebook_file)[0]

        print(f"\nRunning notebook: {notebook_file}")
        pm.execute_notebook(notebook, output_path=f"{script_dir}/../rna_analysis/{notebook_name}_out.ipynb", kernel_name='sctoolbox', log_level="DEBUG", report_mode=True, cwd=notebook_dir)

elif notebook_type == "ATAC":
    # Run ATAC notebooks
    notebook_dir = script_dir + "/../atac_analysis/notebooks/"
    atac_notebooks = sorted(glob.glob(notebook_dir + "*.ipynb"), key=key)  # sort as glob output is not ordered
    atac_notebooks = [nb for nb in atac_notebooks]
    print("\n\n")
    print("--------------------------------------- ATAC ---------------------------------------")
    print(f"Notebook directory: {notebook_dir}")
    print(f"Notebooks: {[os.path.basename(f) for f in atac_notebooks]}")
    print("------------------------------------------------------------------------------------")
    print("\n\n")

    for notebook in atac_notebooks:
        # get filename
        notebook_file = os.path.basename(notebook)
        notebook_name = os.path.splitext(notebook_file)[0]

        print(f"\nRunning notebook: {os.path.basename(notebook)}")
        pm.execute_notebook(notebook, output_path=f"{script_dir}/../atac_analysis/{notebook_name}_out.ipynb", kernel_name='sctoolbox', log_level="DEBUG", report_mode=True, cwd=notebook_dir)

else:
    raise ValueError(f"Unknown notebook type. Expected 'RNA' or 'ATAC' got {notebook_type}.")
