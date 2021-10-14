from logging import Formatter, handlers, StreamHandler, getLogger, DEBUG
from pathlib import Path
import os

class Logger:
    def __init__(self, name=__name__, log_dir: str = '../log/'):
        self.logger = getLogger(name)
        self.logger.setLevel(DEBUG)
        formatter = Formatter("[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s")

        #stdout
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        #file
        log_dir = Path('../log/')
        os.makedirs(log_dir)
        log_file = log_dir / 'log.log'
        handler = handlers.RotatingFileHandler(filename=log_file,
                                                maxBytes=1048756,
                                                backupCount=3)
        handler.setLevel(DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
