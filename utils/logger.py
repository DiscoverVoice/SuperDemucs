import sys
import logging

class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n':
            self.buffer += message
            if '\n' in self.buffer:
                self.flush()

    def flush(self):
        self.level(self.buffer.strip())
        self.buffer = ''


def stdio2logs():
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)