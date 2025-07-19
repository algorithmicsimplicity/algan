import sys
import traceback

from algan.settings.logging_defaults import LOGGING_DEFAULTS

class LoggerError(Exception):
    pass

class Logger:
    def __init__(self, output_file_path, verbosity=None):
        self.verbosity = verbosity
        self.file_path = output_file_path
        self.verbosity_levels = {'batching': 0,
                                 'memory': 10,
                                 'rendering': 20,
                                 'max': 100}
        self.current_log_class = None
        with open(output_file_path, 'w') as f:
            f.write(' ')

    def set_class(self, log_class):
        self.current_log_class = log_class
        return self

    def get_verbosity(self):
        return LOGGING_DEFAULTS.verbosity if self.verbosity is None else self.verbosity

    def log_message(self, message):
        v = self.get_verbosity()
        if v is None or self.verbosity_levels[self.current_log_class] > self.verbosity_levels[v]:
            return
        self._write_message(message)

    def _write_message(self, message):
        with open(self.file_path, 'a') as f:
            traceback.print_stack(file=f)
            f.write('\n')
            f.write('*' * 20)
            f.write('\n')
            f.write(message)
            f.write('\n\n')
            f.write('-' * 20)
            f.write('\n\n')


class LoggerManager:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call LoggerManager.instance() instead of LoggerManager().')

    @classmethod
    def reset(cls):
        cls._instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Logger(LOGGING_DEFAULTS.log_file_path)
        return cls._instance
