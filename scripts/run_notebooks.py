import os
import glob
import papermill as pm

# Get location of script
script_dir = os.path.dirname(__file__)
print(f"Script location: {script_dir}")
rna_notebook_path_suffix = "/../rna_analysis/notebooks/"

# Run RNA notebooks
notebook_dir = script_dir + rna_notebook_path_suffix
rna_notebooks = sorted(glob.glob(notebook_dir + "*.ipynb"), key=str.lower)  # sort as glob output is not ordered
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


# Run ATAC notebooks
notebook_dir = script_dir + "/../atac_analysis/notebooks/"
atac_notebooks = sorted(glob.glob(notebook_dir + "*.ipynb"))  # sort as glob output is not ordered
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
