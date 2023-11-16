import logging
import sys

class StreamToLogger:
    """
    Custom stream object that writes to a logger as well as to a file.
    """
    def __init__(self, logger, log_file):
        self.logger = logger
        self.log_file = open(log_file, 'a')

    def write(self, message):
        if message.rstrip() != "":
            self.logger.info(message.rstrip())
            self.log_file.write(message)

    def flush(self):
        self.log_file.flush()

class Logger:
    def __init__(self, log_file, name='main', level=logging.DEBUG, catch_stdout=True):
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create a file handler that logs messages to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Create a console handler that logs messages to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Redirect stdout to the logger
        if catch_stdout: sys.stdout = StreamToLogger(self.logger, log_file)

    def get_logger(self):
        return self.logger