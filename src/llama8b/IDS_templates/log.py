import logging

import sys

 

 

# Custom formatter

class LogFormatter(logging.Formatter):

 

    blue = "\x1b[34;20m"

    yellow = "\x1b[33;20m"

    red = "\x1b[31;20m"

    reset = "\x1b[0m"

 

    err_fmt = red + "[%(asctime)s] ERROR: %(module)s.py: line:%(lineno)d: %(msg)s" + reset

    dbg_fmt = blue + "[%(asctime)s] DEBUG: %(module)s.py: line:%(lineno)d: %(msg)s" + reset

    #info_fmt = "[%(asctime)s] INFO %(msg)s"

    info_fmt = "[%(asctime)s] INFO %(module)s.py: line:%(lineno)d: %(msg)s"

    warn_fmt = yellow + "[%(asctime)s] WARNING: %(module)s.py: line:%(lineno)d: %(msg)s" + reset

 

    def __init__(self):

        super().__init__(

            fmt="[%(asctime)s]  %(levelno)d: %(msg)s", datefmt=None, style="%"

        )

 

    def format(self, record):

 

        # Save the original format configured by the user

        # when the logger formatter was instantiated

        format_orig = self._style._fmt

 

        # Replace the original format with one customized by logging level

        if record.levelno == logging.DEBUG:

            self._style._fmt = LogFormatter.dbg_fmt

 

        elif record.levelno == logging.INFO:

            self._style._fmt = LogFormatter.info_fmt

 

        elif record.levelno == logging.ERROR:

            self._style._fmt = LogFormatter.err_fmt

 

        elif record.levelno == logging.WARNING:

            self._style._fmt = LogFormatter.warn_fmt

 

        # Call the original formatter class to do the grunt work

        result = logging.Formatter.format(self, record)

 

        # Restore the original format configured by the user

        self._style._fmt = format_orig

 

        return result

 

 

__handler = logging.StreamHandler(sys.stdout)

__handler.setFormatter(LogFormatter())

 

logger = logging.Logger("unknown service")  # This will be updated in app.py

logger.addHandler(__handler)