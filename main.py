from common.configs.log_config import configure_log
from experiments.runner import ExperimentRunner
from common.env.cli import get_cli_env_args

import traceback
import logging
import sys


if __name__ == "__main__":

    try:
        configure_log()
        args = get_cli_env_args()
        runner = ExperimentRunner(args)
        runner.run()

    except KeyboardInterrupt:
        sys.exit(130)

    except Exception as e:
        logging.error(
            f"Unexpected Error: {e}\n" +\
            f"*** Stacktrace:\n{traceback.format_exc()}"
        )
