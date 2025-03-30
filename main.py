from common.configs.log_config import configure_log
from common.env_utils.cli import read_cli_env_args
import experiments.runner as experiment_runner

import traceback
import logging
import sys


if __name__ == "__main__":

    try:
        configure_log()
        args = read_cli_env_args()
        experiment_runner.run(args)

    except KeyboardInterrupt:
        sys.exit(130)

    except Exception as e:
        logging.error(
            f"Unexpected Error: {e}\n" +\
            f"*** Stacktrace:\n{traceback.format_exc()}"
        )
