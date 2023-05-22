import os
import glob
import papermill as pm

# Get location of script
script_dir = os.path.dirname(__file__)
print(script_dir)

# Run RNA notebooks
os.chdir(script_dir + "/../rna-notebooks/")
rna_notebooks = glob.glob("*.ipynb")
rna_notebooks = [nb for nb in rna_notebooks if "05" not in nb and "11" not in nb]

for notebook in rna_notebooks:
    print(f"Running notebook: {notebook}")
    pm.execute_notebook(notebook, output_path="out.ipynb", kernel_name='sctoolbox', log_level="INFO", report_mode=True)
