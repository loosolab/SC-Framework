import os


class SctoolboxConfig(object):
    """
    Config manager for sctoolbox
    """

    __frozen = False

    def __init__(self,
                 figure_prefix: str = "",  # Prefix for all figures to write
                 adata_input_prefix: str = "",  # Prefix for all adata objects to read
                 adata_output_prefix: str = "",   # Prefix for all adata objects to write
                 threads: int = 4,  # default number of threads to use when multiprocessing is available
                 create_dirs: bool = True  # create output directories if they do not exist
                 ):

        self.create_dirs = create_dirs  # must be set first to avoid error when creating directories

        # Save all parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
        self._freeze()  # Freeze the class; no new attributes can be added

    def _freeze(self):
        """ Set __frozen to True, disallowing new attributes to be added """
        self.__frozen = True

    def __setattr__(self, key, value):
        if self.__frozen and not hasattr(self, key):
            valid_parameters = [key for key in self.__dict__ if not key.startswith("_")]
            raise TypeError(f"'{key}' is not a valid setting for sctoolbox. Parameter options are: {valid_parameters}")

        # Validate and set parameter
        if key == "threads":
            self._validate_int(value)
        elif key == "create_dirs":
            self._validate_bool(value)
        elif key in ["figure_prefix", "adata_output_prefix"]:
            self._validate_prefix(value)
        object.__setattr__(self, key, value)

    def _validate_prefix(self, prefix: str):

        # get directory of prefix
        dirname = os.path.dirname(prefix)
        dirname = "./" if dirname == "" else dirname

        # create directory if it does not exist
        self._create_dir(dirname)

    def _validate_int(self, integer: int):
        if not isinstance(integer, int):
            raise TypeError("Parameter must be of type int.")

    def _validate_bool(self, boolean: bool):
        if not isinstance(boolean, bool):
            raise TypeError("Parameter must be of type bool.")

    def _create_dir(self, dirname: str):
        """ Create a directory if it does not exist yet """

        if self.create_dirs is True:
            if not os.path.exists(dirname):
                os.makedirs(dirname)  # creates directory and all parent directories
                print("Created directory: " + dirname)


settings = SctoolboxConfig()
