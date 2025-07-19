from dataclasses import dataclass

__all__ = ['LOGGING_DEFAULTS']

@dataclass
class LogingDefaults:
    log_file_path: str = 'algan_logs.txt'
    verbosity: str = None

LOGGING_DEFAULTS = LogingDefaults()
