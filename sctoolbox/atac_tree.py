import sctoolbox.generalized_tree as generalized_tree
import os

class ATAC_tree(generalized_tree.Tree):
    '''
    Sub-class of Tree.
    This extends Tree by ATAC related directories.
    To add a path:
    1. add variable below and initialise it with None
    2. setup the path in def setupDir
    3. add the path to the to_build list when it leads to a directory,
     what should be created automatically
    4. add property dekorator and setter 
    '''

    # directories
    _pre_pro_dir = None
    _qc_plots_dir = None

    def setupDir(self):
        '''
        Extends the setupDir() function of the super-class Tree from generalized_tree.py.
        This builds paths to ATAC related directories and calls makeDir() to make these,
        if they are not existing.
        :return:
        '''

        # call super function
        super(ATAC_tree, self).setupDir()

        ################################################################################################################
        # setup ATAC related directories:

        # QC notebook related paths
        self._qc_plots_dir = os.path.join(self.qc_dir, 'plots')
        # ADD ATAC PATHS
        ################################
        # list of directories to build if they are not already existing
        to_build = []
        to_build.append(self._qc_plots_dir)

        self.makeDir(to_build)


    # INPUT RELATED DIRECTORIES
    @property
    def pre_pro_dir(self):
        return self._pre_pro_dir

    @pre_pro_dir.setter
    def pre_pro_dir(self, value):
        self._pre_pro_dir = value

    @property
    def qc_plots_dir(self):
        return self._qc_plots_dir

    @qc_plots_dir.setter
    def qc_plots_dir(self, value):
        self._qc_plots_dir = value

if __name__ == "__main__":

    tree = ATAC_tree()

    tree.processing_dir = "/home/jan/python-workspace/sc-atac/test_processed_dir"

    run = tree.run
    tree.run = "some_test"

    pre_pro_dir = tree.pre_pro_dir
    tree.pre_pro_dir = "some/dir"
    pre_pro_dir = tree.pre_pro_dir