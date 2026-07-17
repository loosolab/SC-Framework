"""Settings for running sctoolbox functions including paths to output directories, logging, verbosity etc."""

import os
import sys
import logging

from beartype import beartype
from beartype.typing import Optional, Tuple


class SctoolboxConfig(object):
    """
    Config manager for sctoolbox.

    Attributes
    ----------
    figure_dir : str, default ""
        Directory to write figures to.
    figure_prefix : str, default ""
        Prefix for all figures to write (within figure_dir).
    table_dir : str, default ""
        Directory to write tables to.
    table_prefix : str, default ""
        Prefix for all tables to write (within table_dir).
    adata_input_dir : str, default ""
        Directory to read adata objects from.
    adata_input_prefix : str, default ""
        Prefix for all adata objects to read (within adata_input_dir).
    adata_output_dir : str, default ""
        Directory to write adata objects to.
    adata_output_prefix : str, default ""
        Prefix for all adata objects to write (within adata_output_dir).
    report_dir : str, default ""
        Directory to collect everything for the final report. Set "" to disable.
    threads : int, default 4
        Default number of threads to use when multiprocessing is available.
    k8s_threads : int, default 1
        Default number of threads to use on a kubernetes cluster.
    create_dirs : bool, default True
        Create output directories if they do not exist.
    verbosity : int, default 1
        Logging verbosity: 0 = error, 1 = info, 2 = debug.
    log_file : str, default None
        Path to log file.
    overwrite_log : bool, default False
        Overwrite log file if it already exists; default is to append.
    dpi : float, default 600
        The resolution in dots per inch used when saving plots.
    report_dpi : float, default 200
        The resolution in dots per inch used when saving report plots.
    repo_dir : str
        The path where additional repositories may be stored.
    """

    __name__ = "SctoolboxConfig"  # added to enable sphinx autodoc
    __frozen: bool = False
    """If True, disallows new attributes to be added.

    :meta private:
    """

    def __init__(self,
                 figure_dir: str = "",            # Directory to write figures to
                 figure_prefix: str = "",         # Prefix for all figures to write (within figure_dir)
                 table_dir: str = "",             # Directory to write tables to
                 table_prefix: str = "",          # Prefix for all tables to write (within table_dir)
                 adata_input_dir: str = "",       # Directory to read adata objects from
                 adata_input_prefix: str = "",    # Prefix for all adata objects to read (within adata_input_dir)
                 adata_output_dir: str = "",      # Directory to write adata objects to
                 adata_output_prefix: str = "",   # Prefix for all adata objects to write (within adata_output_dir)
                 report_dir: str = "",            # Directory to collect everything for the final report.
                 threads: int = 4,                # default number of threads to use when multiprocessing is available
                 k8s_threads: int = 1,            # default number of threads to use on a kubernetes cluster
                 create_dirs: bool = True,        # create output directories if they do not exist
                 verbosity: int = 1,              # logging verbosity: 0 = error, 1 = info, 2 = debug
                 log_file: Optional[str] = None,  # Path to log file
                 overwrite_log: bool = False,     # Overwrite log file if it already exists; default is to append
                 dpi: float = 600,                # The resolution in dots per inch, used to save figures.
                 report_dpi: float = 200,         # The resolution in dots per inch, used for report figures.
                 repo_dir: str = ""               # The path to additional repositores.
                 ) -> None:

        self.create_dirs = create_dirs  # must be set first to avoid error when creating directories

        # Save all parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
        self._freeze()  # Freeze the class; no new attributes can be added

    def reset(self) -> None:
        """Reset all settings to default."""
        self.__init__()

    def _freeze(self) -> None:
        """Set __frozen to True, disallowing new attributes to be added."""
        self.__frozen = True

    def __setattr__(self, key: str, value: object) -> None:
        """Set attribute if it exists in __init__ and is of the correct type.

        Raises
        ------
        ValueError
            If the key is not a valid setting when the class is frozen.
        """

        # Disallow new attributes when frozen
        if self.__frozen and not hasattr(self, key):
            valid_parameters = [key for key in self.__dict__ if not key.startswith("_")]
            raise ValueError(f"'{key}' is not a valid setting for sctoolbox. Parameter options are: {valid_parameters}")

        # Allow internal attributes to be set without validation
        if "__frozen" in key or key == "_logger":
            object.__setattr__(self, key, value)
            return

        # Perform all validations
        value = self._validate_and_process_attribute(key, value)

        # Set the attribute
        object.__setattr__(self, key, value)

        # Setup logger if relevant settings are present
        self._trigger_logger_setup_if_needed(key)

    def _validate_and_process_attribute(self, key: str, value: object) -> object:
        """Validate and process attribute value based on type and key.

        Returns
        -------
        The processed attribute value.

        Raises
        ------
        ValueError
            If verbosity is neither 0, 1 nor 2.
        """
        # Validate log_file specially
        if key == "log_file" and value is not None:
            self._validate_string(value)

        # Validate based on annotations if available
        ann = getattr(self.__init__, "__annotations__", {})
        if key in ann:
            expected = ann[key]
            if expected is int:
                self._validate_int(value)
            elif expected is bool:
                self._validate_bool(value)
            elif expected is str:
                self._validate_string(value)

        # Additional attribute-specific checks
        if key in ["figure_dir", "table_dir", "adata_input_dir", "adata_output_dir", "report_dir", "repo_dir"]:
            value = os.path.join(value, '')  # add trailing slash if not present
            self._create_dir(value)

        if key == "verbosity":
            self._validate_int(value)
            if value not in [0, 1, 2]:
                raise ValueError("Verbosity must be 0, 1 or 2.")

        return value

    def _trigger_logger_setup_if_needed(self, key: str) -> None:
        """Set up logger if all required logging settings are now present."""
        logging_keys = ["verbosity", "overwrite_log", "log_file"]
        if key in logging_keys and all(hasattr(self, k) for k in logging_keys):
            self._setup_logger(verbosity=self.verbosity, log_file=self.log_file, overwrite_log=self.overwrite_log)

    def _validate_string(self, string: str) -> None:
        if not isinstance(string, str):
            raise TypeError("Parameter must be of type str.")

    def _validate_int(self, integer: int) -> None:
        if not isinstance(integer, int):
            raise TypeError("Parameter must be of type int.")

    def _validate_bool(self, boolean: bool) -> None:
        if not isinstance(boolean, bool):
            raise TypeError("Parameter must be of type bool.")

    def _create_dir(self, dirname: str) -> None:
        """Create a directory if it does not exist yet."""

        if dirname == "":  # do not create directory if path is empty
            return

        if self.create_dirs:
            if not os.path.exists(dirname):
                os.makedirs(dirname)  # creates directory and all parent directories
                print("Created directory: " + dirname)

    # Getter / setter for filename prefixes
    @property
    def full_figure_prefix(self) -> None:
        """Combine figure_dir and figure_prefix on the fly to get the full figure prefix."""
        return self.figure_dir + self.figure_prefix   # figure_dir has trailing slash

    @full_figure_prefix.setter
    def full_figure_prefix(self, value: str) -> None:
        raise ValueError("'full_figure_prefix' cannot be set directly. Adjust 'figure_dir' & 'figure_prefix' instead.")

    @property
    def full_table_prefix(self) -> None:
        """Combine table_dir and table_prefix on the fly to get the full table prefix."""
        return self.table_dir + self.table_prefix

    @full_table_prefix.setter
    def full_table_prefix(self, value: str) -> None:
        raise ValueError("'full_table_prefix' cannot be set directly. Adjust 'table_dir' & 'table_prefix' instead.")

    @property
    def full_adata_input_prefix(self) -> str:
        """Combine adata_input_dir and adata_input_prefix on the fly to get the full adata input prefix."""
        return self.adata_input_dir + self.adata_input_prefix

    @full_adata_input_prefix.setter
    def full_adata_input_prefix(self, value: str) -> None:
        raise ValueError("'full_adata_input_prefix' cannot be set directly. Adjust 'adata_input_dir' & 'adata_input_prefix' instead.")

    @property
    def full_adata_output_prefix(self) -> str:
        """Combine adata_output_dir and adata_output_prefix on the fly to get the full adata output prefix."""
        return self.adata_output_dir + self.adata_output_prefix

    @full_adata_output_prefix.setter
    def full_adata_output_prefix(self, value: str) -> None:
        raise ValueError("'full_adata_output_prefix' cannot be set directly. Adjust 'adata_output_dir' & 'adata_output_prefix' instead.")

    def _setup_logger(self, verbosity: int = None, log_file: str = None, overwrite_log: bool = False) -> None:
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
        level, formatter = self._get_logger_level_and_formatter(verbosity, simple_formatter, debug_formatter)

        # Setup logger
        self._logger = logging.getLogger("sctoolbox")
        self._logger.setLevel(logging.DEBUG)  # always log everything to logger, handlers will filter
        self._logger.handlers = []  # remove any existing handlers
        self._logger.propagate = False  # ensure no other handlers are used

        # Add stream handler
        H = logging.StreamHandler(sys.stdout)
        H.setLevel(level)  # level from verbosity
        H.setFormatter(formatter)
        self._logger.addHandler(H)

        # Add file handler if chosen
        if log_file is not None:
            self._setup_file_handler(log_file, overwrite_log, debug_formatter)

    def _get_logger_level_and_formatter(self, verbosity: int, simple_formatter: logging.Formatter, debug_formatter: logging.Formatter) -> Tuple[int, logging.Formatter]:
        """Get logger level and formatter based on verbosity.

        Returns
        -------
        tuple
            A tuple of (logging_level, formatter).
        """
        if verbosity == 0:
            return logging.ERROR, simple_formatter
        elif verbosity == 1:
            return logging.INFO, simple_formatter
        else:  # verbosity == 2
            return logging.DEBUG, debug_formatter

    def _setup_file_handler(self, log_file: str, overwrite_log: bool, debug_formatter: logging.Formatter) -> None:
        """Set up file handler for logger.

        Raises
        ------
        ValueError
            If the logfile cannot be created or written.
        """
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

    def close_logfile(self) -> None:
        """Close all open filehandles of logger."""
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    @property
    def logger(self) -> logging.Logger:
        """Return logger object."""
        return self._logger

    @beartype
    def settings_from_config(self: object, config_file: str, key: Optional[str] = None) -> None:
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
        import yaml  # import here so it can be removed from the build-system requirements

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

    def get_threads(self) -> int:
        """Return the number of threads. Either self.threads or self.k8s_threads.

        Returns
        -------
        int
            The number of threads (k8s_threads if running on Kubernetes, otherwise threads).
        """
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            return self.k8s_threads
        return self.threads


settings = SctoolboxConfig()
