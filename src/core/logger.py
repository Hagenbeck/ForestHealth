import os
import sys
import threading
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path


class LogSegment(Enum):
    """Enum defining logging segments for different modules and components"""
    CORE = "CORE"
    THREADS = "THREADS"
    DATA_DOWNLOAD = "DATA_DOWNLOAD"
    DATA_SOURCING = "DATA_SOURCING"
    DATA_PROCESSING = "DATA_PROCESSING"
    DEM_PROCESSOR = "DEM_PROCESSOR"
    CLUSTERING = "CLUSTERING"
    SENTINEL_API = "SENTINEL_API"
    GEOMETRY_TOOLKIT = "GEOMETRY_TOOLKIT"


class Logger:
    """Singleton class to log our workflow"""

    _instance = None
    _logs = []
    _logfile = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    def get_instance(cls):
        """Function to return instance of singleton class

        Returns:
            Logger: Singleton instance of class
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._init_logfile()
            cls._instance._setup_hooks()
        return cls._instance

    def _init_logfile(self):
        """Function to initialize the log-file and create the log-folder if it doesn't exist already"""
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        root_path = Path.cwd().resolve()
        log_dir = root_path / "logs"
        os.makedirs(log_dir, exist_ok=True)

        self._logfile = log_dir / f"session_{timestamp}.log"

    def _setup_hooks(self):
        """Function to setup the exception hooks"""
        sys.excepthook = self._handle_exception
        threading.excepthook = self._thread_excepthook

    @staticmethod
    def get_timestamp() -> str:
        """Function to get logging timestamp

        Returns:
            str: string representation of Hour:Minute:Seconds.Milliseconds
        """
        return datetime.now().strftime("%H:%M:%S.%f")

    def add_log(self, line: str):
        self._logs.append(line)
        print(line)

    def info(self, segment: LogSegment, message: str):
        """Method to log an info message

        Args:
            segment (LogSegment): Enum value identifying the source
            message (str): str representation of message
        """
        line = f"{Logger.get_timestamp()}: [INFO] {segment.value} {message}\n"
        self.add_log(line)

    def warning(self, segment: LogSegment, message: str):
        """Method to log a warning message

        Args:
            segment (LogSegment): Enum value identifying the source
            message (str): str representation of message
        """
        line = f"{Logger.get_timestamp()}: [WARNING] {segment.value} {message}\n"
        self.add_log(line)

    def error(self, segment: LogSegment, message: str):
        """Method to log an error message and print it to the terminal

        Args:
            segment (LogSegment): Enum value identifying the source
            message (str): str representation of message
        """
        print("Error " + segment.value + ": " + message)
        line = f"{Logger.get_timestamp()}: [ERROR] {segment.value} {message}\n"
        self.add_log(line)

    def _flush_logs(self):
        """write logs to the log-file"""
        if self._logfile and self._logs:
            with open(self._logfile, "a", encoding="utf-8") as f:
                f.write("".join(self._logs))
            self._logs = []

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Function to handle exceptions of the main thread

        Args:
            exc_type (type[BaseException]): class of the exception that was raised
            exc_value (BaseException): exception instance that was raised
            exc_traceback (Traceback): Traceback of the exception
        """
        self.error(LogSegment.CORE, "Uncaught exception:")
        self.error(
            LogSegment.CORE,
            "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
        )
        self._flush_logs()

    def _thread_excepthook(self, args):
        """Function to handle exceptions of threading sub-threads

        Args:
            args: args of a threading exception
        """
        self.error(
            LogSegment.THREADS,
            f"Unhandled exception in thread {args.thread.name}: {args.exc_value}",
        )
        self.error(
            LogSegment.THREADS,
            "".join(
                traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback
                )
            ),
        )
        self._flush_logs()
