from common.cli.env_args import CliEnvArgs
from common.configs import LogConfig
from experiments import ExperimentRunner

import traceback
import logging
import sys


if __name__ == "__main__":

    try:
        LogConfig.configure()
        args = CliEnvArgs.get_args()
        runner = ExperimentRunner(args)
        runner.run()

    except KeyboardInterrupt:
        sys.exit(130)

    except Exception as e:
        logging.error(
            f"Unexpected Error: {e}\n" +\
            f"*** Stacktrace:\n{traceback.format_exc()}"
        )