"""Settings for running sctoolbox functions including paths to output directories, logging, verbosity etc."""

import os
import yaml
import sys
import logging

from beartype import beartype
from beartype.typing import Optional


class SctoolboxConfig(object):
    """
    Config manager for sctoolbox.

    Attributes
    ----------
    __frozen : bool
        If True, disallows new attributes to be added.

    Parameters
    ----------
    figure_dir : str
        Directory to write figures to, default "".
    figure_prefix : str
        Prefix for all figures to write (within figure_dir), default "".
    table_dir : str
        Directory to write tables to, default "".
    table_prefix : str
        Prefix for all tables to write (within table_dir), default "".
    adata_input_dir : str
        Directory to read adata objects from, default "".
    adata_input_prefix : str
        Prefix for all adata objects to read (within adata_input_dir), default "".
    adata_output_dir : str
        Directory to write adata objects to, default "".
    adata_output_prefix : str
        Prefix for all adata objects to write (within adata_output_dir), default "".
    report_dir : str
        Directory to collect everything for the final report, default "" to disable.
    threads : int
        Default number of threads to use when multiprocessing is available, default 4.
    create_dirs : bool
        Create output directories if they do not exist, default True.
    verbosity : int
        Logging verbosity: 0 = error, 1 = info, 2 = debug, default 1.
    log_file : str
        Path to log file, default None.
    overwrite_log : bool
        Overwrite log file if it already exists; default is to append, default False.
    """

    __frozen: bool = False

    def __init__(self,
                 figure_dir: str = "",           # Directory to write figures to
                 figure_prefix: str = "",        # Prefix for all figures to write (within figure_dir)
                 table_dir: str = "",            # Directory to write tables to
                 table_prefix: str = "",         # Prefix for all tables to write (within table_dir)
                 adata_input_dir: str = "",      # Directory to read adata objects from
                 adata_input_prefix: str = "",   # Prefix for all adata objects to read (within adata_input_dir)
                 adata_output_dir: str = "",     # Directory to write adata objects to
                 adata_output_prefix: str = "",  # Prefix for all adata objects to write (within adata_output_dir)
                 report_dir: str = "",           # Directory to collect everything for the final report.
                 threads: int = 4,               # default number of threads to use when multiprocessing is available
                 create_dirs: bool = True,       # create output directories if they do not exist
                 verbosity: int = 1,             # logging verbosity: 0 = error, 1 = info, 2 = debug
                 log_file: str = None,           # Path to log file
                 overwrite_log: bool = False,    # Overwrite log file if it already exists; default is to append
                 ):

        self.create_dirs = create_dirs  # must be set first to avoid error when creating directories

        # Save all parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
        self._freeze()  # Freeze the class; no new attributes can be added

    def reset(self):
        """Reset all settings to default."""
        self.__init__()

    def _freeze(self):
        """Set __frozen to True, disallowing new attributes to be added."""
        self.__frozen = True

    def __setattr__(self, key, value):
        """Set attribute if it exists in __init__ and is of the correct type."""

        if self.__frozen and not hasattr(self, key):
            valid_parameters = [key for key in self.__dict__ if not key.startswith("_")]
            raise ValueError(f"'{key}' is not a valid setting for sctoolbox. Parameter options are: {valid_parameters}")

        # Validate parameter types
        if "__frozen" in key:  # allow __frozen to be set without checking
            pass
        elif key == "_logger":
            pass  # allow logger to be set without checking

        elif key == "log_file":
            if value is not None:
                self._validate_string(value)

        elif self.__init__.__annotations__[key] == int:
            self._validate_int(value)

        elif self.__init__.__annotations__[key] == bool:
            self._validate_bool(value)

        elif self.__init__.__annotations__[key] == str:
            self._validate_string(value)

        # Additionally check specific attributes for validity
        if key in ["figure_dir", "table_dir", "adata_input_dir", "adata_output_dir", "report_dir"]:
            value = os.path.join(value, '')  # add trailing slash if not present
            self._create_dir(value)

        elif key == "verbosity":
            self._validate_int(value)
            if value not in [0, 1, 2]:
                raise ValueError("Verbosity must be 0, 1 or 2.")

        object.__setattr__(self, key, value)

        # Setup logger if certain parameters are changed
        logging_keys = ["verbosity", "overwrite_log", "log_file"]
        if key in logging_keys:
            if all([hasattr(self, key) for key in logging_keys]):  # only set logger if all keys were set at least once. This avoids setting the logger when the individual keys are initialized for the first time
                self._setup_logger(verbosity=self.verbosity, log_file=self.log_file, overwrite_log=self.overwrite_log)

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
        """Create a directory if it does not exist yet."""

        if dirname == "":  # do not create directory if path is empty
            return

        if self.create_dirs:
            if not os.path.exists(dirname):
                os.makedirs(dirname)  # creates directory and all parent directories
                print("Created directory: " + dirname)

    # Getter / setter for filename prefixes
    @property
    def full_figure_prefix(self):
        """Combine figure_dir and figure_prefix on the fly to get the full figure prefix."""
        return self.figure_dir + self.figure_prefix   # figure_dir has trailing slash

    @full_figure_prefix.setter
    def full_figure_prefix(self, value):
        raise ValueError("'full_figure_prefix' cannot be set directly. Adjust 'figure_dir' & 'figure_prefix' instead.")

    @property
    def full_table_prefix(self):
        """Combine table_dir and table_prefix on the fly to get the full table prefix."""
        return self.table_dir + self.table_prefix

    @full_table_prefix.setter
    def full_table_prefix(self, value):
        raise ValueError("'full_table_prefix' cannot be set directly. Adjust 'table_dir' & 'table_prefix' instead.")

    @property
    def full_adata_input_prefix(self):
        """Combine adata_input_dir and adata_input_prefix on the fly to get the full adata input prefix."""
        return self.adata_input_dir + self.adata_input_prefix

    @full_adata_input_prefix.setter
    def full_adata_input_prefix(self, value):
        raise ValueError("'full_adata_input_prefix' cannot be set directly. Adjust 'adata_input_dir' & 'adata_input_prefix' instead.")

    @property
    def full_adata_output_prefix(self):
        """Combine adata_output_dir and adata_output_prefix on the fly to get the full adata output prefix."""
        return self.adata_output_dir + self.adata_output_prefix

    @full_adata_output_prefix.setter
    def full_adata_output_prefix(self, value):
        raise ValueError("'full_adata_output_prefix' cannot be set directly. Adjust 'adata_output_dir' & 'adata_output_prefix' instead.")

    def _setup_logger(self, verbosity: int = None, log_file: str = None, overwrite_log: bool = False):
        """Set up logger on the basis of the verbosity level."""

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
                    if overwrite_log:
                        self._logger.warning(f"Log file '{log_file}' already exists. The file will be overwritten since 'overwrite_log' is set to True.")
                    else:
                        self._logger.warning(f"Log file '{log_file}' already exists. Logging messages will be appended to file. Set overwrite_log=True to overwrite the file.")
                else:
                    raise ValueError(f"Log file '{log_file}' already exists but cannot be written to. Please choose a different file name.")
            else:
                parent_dir = os.path.dirname(log_file)
                parent_dir = "." if parent_dir == "" else parent_dir  # if log_file is in current directory, parent_dir is empty
                self._create_dir(parent_dir)  # create parent directory if it does not exist
                if not os.access(parent_dir, os.W_OK):
                    raise ValueError(f"Log file '{log_file}' cannot be created. Please check that the directory exists and is writable.")

            mode = "w" if overwrite_log else "a"
            F = logging.FileHandler(log_file, mode=mode)
            F.setLevel(logging.DEBUG)        # always log all messages to file
            F.setFormatter(debug_formatter)  # always use debug formatter for file handler
            self._logger.addHandler(F)

    def close_logfile(self):
        """Close all open filehandles of logger."""
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    @property
    def logger(self):
        """Return logger object."""
        return self._logger

    @beartype
    def settings_from_config(self: object, config_file: str, key: Optional[str] = None):
        """
        Set settings from a config file in yaml format. Settings are set directly in sctoolbox.settings.

        Parameters
        ----------
        config_file : str
            Path to the config file.
        key : Optional[str], default None
            If given, get settings for a specific key.

        Raises
        ------
        KeyError
            If key is not found in config file.
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
        preferred_order = ["overwrite_log", "log_file"]  # set overwrite_log before log_file to prevent "log file already exists" warning
        for key, value in sorted(config_dict.items(), key=lambda x: preferred_order.index(x[0]) if x[0] in preferred_order else 999):
            setattr(settings, key, value)


settings = SctoolboxConfig()
