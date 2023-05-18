import os
import yaml


class SctoolboxConfig(object):
    """
    Config manager for sctoolbox
    """

    __frozen = False

    def __init__(self,
                 figure_dir: str = "",           # Directory to write figures to
                 figure_prefix: str = "",        # Prefix for all figures to write (within figure_path)
                 adata_input_dir: str = "",      # Directory to read adata objects from
                 adata_input_prefix: str = "",   # Prefix for all adata objects to read (within adata_input_path)
                 adata_output_dir: str = "",     # Directory to write adata objects to
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
        elif key in ["figure_dir", "adata_input_dir", "adata_output_dir"]:
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

        if self.create_dirs:
            if not os.path.exists(dirname):
                os.makedirs(dirname)  # creates directory and all parent directories
                print("Created directory: " + dirname)

    # Getter / setter for filename prefixes
    @property
    def full_figure_prefix(self):
        """ Combine figure_dir and figure_prefix on the fly to get the full figure prefix """
        return self.figure_dir + self.figure_prefix   # figure_dir has trailing slash

    @full_figure_prefix.setter
    def full_figure_prefix(self, value):
        raise ValueError("'full_figure_prefix' cannot be set directly. Adjust 'figure_dir' & 'figure_prefix' instead.")

    @property
    def full_adata_input_prefix(self):
        """ Combine adata_input_dir and adata_input_prefix on the fly to get the full adata input prefix """
        return self.adata_input_dir + self.adata_input_prefix

    @full_adata_input_prefix.setter
    def full_adata_input_prefix(self, value):
        raise ValueError("'full_adata_input_prefix' cannot be set directly. Adjust 'adata_input_dir' & 'adata_input_prefix' instead.")

    @property
    def full_adata_output_prefix(self):
        """ Combine adata_output_dir and adata_output_prefix on the fly to get the full adata output prefix """
        return self.adata_output_dir + self.adata_output_prefix

    @full_adata_output_prefix.setter
    def full_adata_output_prefix(self, value):
        raise ValueError("'full_adata_output_prefix' cannot be set directly. Adjust 'adata_output_dir' & 'adata_output_prefix' instead.")


def settings_from_config(config_file, key=None):
    """
    Set settings from a config file in yaml format.

    Parameters
    ----------
    config_file : str
        Path to the config file.
    key : str, optional
        If given, get settings for a specific key.

    Returns
    -------
    None
        Settings are set in sctoolbox.settings.
    """

    # Read yaml file
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    if key is not None:
        try:
            config_dict = config_dict[key]
        except KeyError:
            raise KeyError(f"Key {key} not found in config file {config_file}")

    # Set settings
    for key, value in config_dict.items():
        setattr(settings, key, value)


settings = SctoolboxConfig()
