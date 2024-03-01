import os
import glob
import papermill as pm

# Get location of script
script_dir = os.path.dirname(__file__)
print(script_dir)
rna_notebook_path_suffix = "/../rna_analysis/notebooks/"

# Run RNA notebooks
notebook_dir = script_dir + rna_notebook_path_suffix
rna_notebooks = sorted(glob.glob(notebook_dir + "*.ipynb"))  # sort as glob output is not ordered
rna_notebooks = [nb for nb in rna_notebooks if "05" not in nb and "14" not in nb] # 05 not tested yet; 14 cannot be tested due to missing input file (vdata has to be added in the future).
print(rna_notebooks)

for notebook in rna_notebooks:
    # get filename
    notebook_file = os.path.basename(notebook)
    notebook_name = os.path.splitext(notebook_file)[0]

    print(f"Running notebook: {notebook_file}")

    # create log files
    std_out_file = open(f"{script_dir}/../rna_analysis/{notebook_name}_std_out.txt", "w")
    std_err_file = open(f"{script_dir}/../rna_analysis/{notebook_name}_std_err.txt", "w")

    pm.execute_notebook(notebook, output_path=f"{script_dir}/../rna_analysis/{notebook_name}_out.ipynb", kernel_name='sctoolbox', log_level="DEBUG", report_mode=True, cwd=notebook_dir, stdout_file=std_out_file, stderr_file=std_err_file)
    std_out_file.close()
    std_err_file.close()
# Run ATAC notebooks
notebook_dir = script_dir + "/../atac_analysis/notebooks/"
atac_notebooks = sorted(glob.glob(notebook_dir + "*.ipynb"))  # sort as glob output is not ordered
atac_notebooks = [nb for nb in atac_notebooks if "05" not in nb and "06" not in nb]  # 05 is not tested yet; 06 uses the output of 05
print(atac_notebooks)

for notebook in atac_notebooks:
    print(f"Running notebook: {notebook}")
    pm.execute_notebook(notebook, output_path="out.ipynb", kernel_name='sctoolbox', log_level="INFO", report_mode=True, cwd=notebook_dir)

# Run general notebooks
notebook_dir = script_dir + "/../general_notebooks/"
general_notebooks = glob.glob(notebook_dir + "*.ipynb")
print(general_notebooks)

for notebook in general_notebooks:
    print(f"Running notebook: {notebook}")
    pm.execute_notebook(notebook, output_path="out.ipynb", kernel_name='sctoolbox', log_level="INFO", report_mode=True, cwd=script_dir + rna_notebook_path_suffix)
