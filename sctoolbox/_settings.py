import os
import yaml
import sys
import logging


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
                 create_dirs: bool = True,  # create output directories if they do not exist
                 verbosity: int = 1,             # 0 = error, 1 = info, 2 = debug
                 log_file: str = None,           # Path to log file
                 ):
        """
        """
        self.create_dirs = create_dirs  # must be set first to avoid error when creating directories

        # Save all parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
        self._freeze()  # Freeze the class; no new attributes can be added

    def reset(self):
        """ Reset all settings to default """
        self.__init__()

    def _freeze(self):
        """ Set __frozen to True, disallowing new attributes to be added """
        self.__frozen = True

    def __setattr__(self, key, value):
        """ Set attribute if it exists in __init__ and is of the correct type """

        if self.__frozen and not hasattr(self, key):
            valid_parameters = [key for key in self.__dict__ if not key.startswith("_")]
            raise ValueError(f"'{key}' is not a valid setting for sctoolbox. Parameter options are: {valid_parameters}")

        # Validate and set parameter
        if "__frozen" in key:  # allow __frozen to be set without checking
            pass
        elif key == "_logger":
            pass  # allow logger to be set without checking

        elif key in ["figure_dir", "adata_input_dir", "adata_output_dir"]:
            value = os.path.join(value, '')  # add trailing slash if not present
            self._validate_string(value)
            self._create_dir(value)

        elif key == "verbosity":
            self._setup_logger(verbosity=value)

        elif key == "log_file":
            if value is not None:
                self._validate_string(value)
                self._setup_logger(log_file=value)  # re-setup logger with new log file

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

    def _setup_logger(self, verbosity: int = None, log_file: str = None):
        """ Set up logger on the basis of the verbosity level """

        # Use current settings if no new settings are provided
        if log_file is None:
            log_file = self.log_file if hasattr(self, "log_file") else None  # log_file is not set at the time of first logger creation (triggered by setting verbosity)
        if verbosity is None:
            verbosity = self.verbosity

        # Setup formatting of handler
        simple_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        debug_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s", "%d-%m-%Y %H:%M:%S")

        # Select logger level
        if verbosity == 0:
            level = logging.ERROR
            formatter = simple_formatter
        elif verbosity == 1:
            level = logging.INFO
            formatter = simple_formatter
        elif verbosity == 2:
            level = logging.DEBUG
            formatter = debug_formatter

        # Setup logger
        self._logger = logging.getLogger("sctoolbox")
        self._logger.setLevel(logging.DEBUG)  # always log everything to logger, handlers will filter
        self._logger.handlers = []  # remove any existing handlers

        # Add stream handler
        H = logging.StreamHandler(sys.stdout)
        H.setLevel(level)  # level from verbosity
        H.setFormatter(formatter)
        self._logger.addHandler(H)

        # Add file handler if chosen
        if log_file is not None:
            if os.path.exists(log_file):
                if os.access(log_file, os.W_OK):
                    self._logger.warning("Log file already exists. Logging messages will be appended to file.")
                else:
                    raise ValueError("Log file already exists but cannot be written to. Please choose a different file name.")
            else:
                parent_dir = os.path.dirname(log_file)
                parent_dir = "." if parent_dir == "" else parent_dir  # if log_file is in current directory, parent_dir is empty
                if not os.access(parent_dir, os.W_OK):
                    raise ValueError("Log file cannot be created. Please check that the directory exists and is writable.")

            F = logging.FileHandler(log_file)
            F.setLevel(logging.DEBUG)        # always log all messages to file
            F.setFormatter(debug_formatter)  # always use debug formatter for file handler
            self._logger.addHandler(F)

    @property
    def logger(self):
        return self._logger


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
