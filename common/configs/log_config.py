from typing import Optional

import logging
import sys


class LogConfig:

    @staticmethod
    def configure(
        log_format_str: Optional[str] = None,
        datefmt: Optional[str] = "%d-%m-%Y %H:%M:%S",
        level: Optional[int] = logging.INFO
    ) -> None:

        if log_format_str is None:
            log_format_str =\
                "%(asctime)s %(levelname)s:%(filename)s:%(funcName)s: %(message)s"

        logging.basicConfig(
            format=log_format_str,
            datefmt=datefmt,
            level=level,
            stream=sys.stdout
        )
