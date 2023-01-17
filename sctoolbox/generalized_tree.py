import os
from pathlib import Path


class Tree:
    '''
    Super class to handle sc-framework related directories.
    This is extended by the sub-classes ATAC_tree and RNA_tree
    '''

    def __init__(self):
        '''
        Initializes path variables with None
        Returns
        -------
        None.

        '''

        # Name of the experiment or Sample
        self._run = None

        # Output related Directories
        self._processing_dir = None
        self._processed_run = None

        # 1. Assemble Adata
        self._assemble_dir = None
        self._assembled_anndata_dir = None
        self._assembled_anndata = None

        # 2. QC
        self._qc_dir = None
        self._qc_anndata_dir = None
        self._qc_anndata = None
        self._qc_plots = None

        # 3. Norm, correction and comparison
        self._norm_correction_dir = None
        self._norm_correction_anndata_dir = None
        self._norm_correction_anndata = None
        self._norm_correction_plots = None

        # 4. Clustering
        self._clustering_dir = None
        self._clustering_anndata_dir = None
        self._clustering_anndata = None
        self._clustering_plots = None

        # 5. Annotation
        self._annotation_dir = None
        self._annotation_anndata_dir = None
        self._annotation_anndata = None
        self._annotation_plots = None

        self._complete_report_dir = None

    def makeDir(self, to_build):
        '''
        Method to make directories given by a list of paths
        :param to_build: list of str
        :return: None
        '''

        for path in to_build:
            if path is not None and not os.path.isdir(path):
                try:
                    Path(path).mkdir(parents=True)
                    print(path + ': NEWLY SETUP')

                except Exception as e:
                    print(e)
        print('all directories existing')

    def setupDir(self):
        '''
        Builds paths to general directories,
        after all necessary inputs are given(checking takes place in the setter methods)
        calls makeDir to make the directories
        :return: None
        '''

        # build run related paths:
        self._processed_run_dir = os.path.join(self._processing_dir, self._run)

        # build notebooks paths
        self._assemble_dir = os.path.join(self._processed_run_dir, 'assembling')
        self._assembled_anndata_dir = os.path.join(self._assemble_dir, 'anndata')
        self._assembled_anndata = os.path.join(self._assembled_anndata_dir, self._run + '.h5ad')

        self._qc_dir = os.path.join(self._processed_run_dir, 'qc')
        self._qc_anndata_dir = os.path.join(self._qc_dir, 'anndata')
        self._qc_anndata = os.path.join(self._qc_anndata_dir, self._run + '.h5ad')
        self._qc_plots = os.path.join(self._qc_dir, 'plots')

        self._norm_correction_dir = os.path.join(self._processed_run_dir, 'norm_correction')
        self._norm_correction_anndata_dir = os.path.join(self._norm_correction_dir, 'anndata')
        self._norm_correction_anndata = os.path.join(self._norm_correction_anndata_dir, self._run + '.h5ad')
        self._norm_correction_plots = os.path.join(self._norm_correction_dir, 'plots')

        self._clustering_dir = os.path.join(self._processed_run_dir, 'clustering')
        self._clustering_anndata_dir = os.path.join(self._clustering_dir, 'anndata')
        self._clustering_anndata = os.path.join(self._clustering_anndata_dir, self.run + '.h5ad')
        self._clustering_plots = os.path.join(self._clustering_dir, 'plots')

        self._annotation_dir = os.path.join(self._processed_run_dir, 'annotation')
        self._annotation_anndata_dir = os.path.join(self._annotation_dir, 'anndata')
        self._annotation_anndata = os.path.join(self._annotation_anndata_dir, self._run + '.h5ad')
        self._annotation_plots = os.path.join(self._annotation_dir, 'plots')


        # complete report dir
        self._complete_report_dir = os.path.join(self._processed_run_dir, 'complete_report')

        # list of directories to build if they are not already existing
        to_build = []
        to_build.append(self.assemble_dir)
        to_build.append(self._assembled_anndata_dir)
        to_build.append(self._qc_dir)
        to_build.append(self._qc_anndata_dir)
        to_build.append(self._qc_plots)
        to_build.append(self._norm_correction_dir)
        to_build.append(self._norm_correction_anndata_dir)
        to_build.append(self._norm_correction_plots)
        to_build.append(self._clustering_dir)
        to_build.append(self._clustering_anndata_dir)
        to_build.append(self._clustering_plots)
        to_build.append(self._annotation_dir)
        to_build.append(self._annotation_anndata_dir)
        to_build.append(self._annotation_plots)
        to_build.append(self._complete_report_dir)

        self.makeDir(to_build)

########################################################################################################################
    # CLASS PROPERTIES (GETTER AND SETTER)

    @property
    def run(self):
        return self._run

    @run.setter
    def run(self, value):
        self._run = value
        # call setupDir if the processing directorie is defined
        if self._processing_dir is not None:
            self.setupDir()
        else:
            print("Warning: processing_dir is None")

    @property
    def processing_dir(self):
        return self._processing_dir

    @processing_dir.setter
    def processing_dir(self, value):
        self._processing_dir = value
        # call setupDir if the run name is defined
        if self._run is not None:
            self.setupDir()
        else:
            print("Warning: run is None")

    @property
    def processed_run_dir(self):
        return self._processed_run_dir

    @processed_run_dir.setter
    def processed_run_dir(self, value):
        self._processed_run_dir = value

    @property
    def assemble_dir(self):
        return self._assemble_dir

    @assemble_dir.setter
    def assemble_dir(self, value):
        self._assemble_dir = value

    @property
    def assembled_anndata_dir(self):
        return self._assembled_anndata_dir

    @assembled_anndata_dir.setter
    def assembled_anndata_dir(self, value):
        self._assembled_anndata_dir = value

    @property
    def assembled_anndata(self):
        return self._assembled_anndata

    @assembled_anndata.setter
    def assembled_anndata(self, value):
        self._assembled_anndata = value

    @property
    def qc_dir(self):
        return self._qc_dir

    @qc_dir.setter
    def qc_dir(self, value):
        self._qc_dir = value

    @property
    def qc_anndata_dir(self):
        return self._qc_anndata_dir

    @qc_anndata_dir.setter
    def qc_anndata_dir(self, value):
        self._qc_anndata_dir = value

    @property
    def qc_anndata(self):
        return self._qc_anndata

    @qc_anndata.setter
    def qc_anndata(self, value):
        self._qc_anndata = value

    @property
    def qc_plots(self):
        return self._qc_plots

    @qc_plots.setter
    def qc_plots(self, value):
        self._qc_plots = value

    @property
    def norm_correction_dir(self):
        return self._norm_correction_dir

    @norm_correction_dir.setter
    def norm_correction_dir(self, value):
        self._norm_correction_dir = value

    @property
    def norm_correction_anndata_dir(self):
        return self._norm_correction_anndata_dir

    @norm_correction_anndata_dir.setter
    def norm_correction_anndata_dir(self, value):
        self._norm_correction_anndata_dir = value

    @property
    def norm_correction_anndata(self):
        return self._norm_correction_anndata

    @norm_correction_anndata.setter
    def norm_correction_anndata(self, value):
        self._norm_correction_anndata = value

    @property
    def norm_correction_plots(self):
        return self._norm_correction_plots

    @norm_correction_plots.setter
    def norm_correction_plots(self, value):
        self._norm_correction_plots = value

    @property
    def clustering_dir(self):
        return self._clustering_dir

    @clustering_dir.setter
    def clustering_dir(self, value):
        self._clustering_dir = value

    @property
    def clustering_anndata_dir(self):
        return self._clustering_anndata_dir

    @clustering_anndata_dir.setter
    def clustering_anndata_dir(self, value):
        self._clustering_anndata_dir = value

    @property
    def clustering_anndata(self):
        return self._clustering_anndata

    @clustering_anndata.setter
    def clustering_anndata(self, value):
        self._clustering_anndata = value

    @property
    def clustering_plots(self):
        return self._clustering_plots

    @clustering_plots.setter
    def clustering_plots(self, value):
        self._clustering_plots = value

    @property
    def annotation_dir(self):
        return self._annotation_dir

    @annotation_dir.setter
    def annotation_dir(self, value):
        self._annotation_dir = value

    @property
    def annotation_anndata_dir(self):
        return self._annotation_anndata_dir

    @annotation_anndata_dir.setter
    def annotation_anndata_dir(self, value):
        self._annotation_anndata_dir = value

    @property
    def annotation_anndata(self):
        return self._annotation_anndata

    @annotation_anndata.setter
    def annotation_anndata(self, value):
        self._annotation_anndata = value

    @property
    def annotation_plots(self):
        return self._annotation_plots

    @annotation_plots.setter
    def annotation_plots(self, value):
        self._annotation_plots = value
        