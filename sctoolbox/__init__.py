from ._version import __version__
from ._settings import settings
from importlib import import_module
import sys

# ---- Ensure backwards compatibility of modules ---- #

old_to_new = {}

# Utility modules
old_util_modules = ["utilities", "creators", "assemblers", "file_converter", "checker"]
old_to_new.update({util: "utils" for util in old_util_modules})

# Tool modules
old_tool_modules = ["analyser", "annotation", "atac", "atac_utils", "bam", "calc_overlap_pct",
                    "celltype_annotation", "custom_celltype_annotation", "marker_genes",
                    "nucleosome_utils", "qc_filter", "receptor_ligand", "multiomics"]
old_to_new.update({tool: "tools" for tool in old_tool_modules})

# Make connection using sys.modules and attributes
sctoolbox_module = import_module("sctoolbox")

for old_module_name, new_module_name in old_to_new.items():
    new_module = import_module("sctoolbox." + new_module_name)

    sys.modules["sctoolbox." + old_module_name] = new_module
    setattr(sctoolbox_module, old_module_name, new_module)  # sets sctoolbox.<module> = sctoolbox.<new_base>.<module>
