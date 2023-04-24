import os


class SctoolboxConfig(object):
    """
    Config manager for sctoolbox
    """

    __frozen = False

    def __init__(self,
                 figure_path: str = "",          # Path to write figures to
                 figure_prefix: str = "",        # Prefix for all figures to write (within figure_path)
                 adata_input_path: str = "",     # Path to read adata objects from
                 adata_input_prefix: str = "",   # Prefix for all adata objects to read (within adata_input_path)
                 adata_output_path: str = "",    # Path to write adata objects to
                 adata_output_prefix: str = "",  # Prefix for all adata objects to write (within adata_output_path)
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
        """ Set attribute if it exists in __init__ and is of the correct type """

        if self.__frozen and not hasattr(self, key):
            valid_parameters = [key for key in self.__dict__ if not key.startswith("_")]
            raise TypeError(f"'{key}' is not a valid setting for sctoolbox. Parameter options are: {valid_parameters}")

        # Validate and set parameter
        if "__frozen" in key:  # allow __frozen to be set without checking
            pass
        elif key in ["figure_path", "adata_input_path", "adata_output_path"]:
            value = os.path.join(value, '')  # add trailing slash if not present
            self._validate_string(value)
            self._create_dir(value)
        elif self.__init__.__annotations__[key] == int:
            self._validate_int(value)
        elif self.__init__.__annotations__[key] == bool:
            self._validate_bool(value)
        elif self.__init__.__annotations__[key] == str:
            self._validate_string(value)

        object.__setattr__(self, key, value)

    def _validate_string(self, string: str):
        if not isinstance(string, str):
            raise TypeError("Parameter must be of type str.")

    def _validate_int(self, integer: int):
        if not isinstance(integer, int):
            raise TypeError("Parameter must be of type int.")

    def _validate_bool(self, boolean: bool):
        if not isinstance(boolean, bool):
            raise TypeError("Parameter must be of type bool.")

    def _create_dir(self, dirname: str):
        """ Create a directory if it does not exist yet """

        if dirname == "":  # do not create directory if path is empty
            return

        if self.create_dirs is True:
            if not os.path.exists(dirname):
                os.makedirs(dirname)  # creates directory and all parent directories
                print("Created directory: " + dirname)


settings = SctoolboxConfig()
