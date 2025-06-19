import logging
import logging.config

from config import LOGGING_CONFIG

# Apply the config
logging.config.dictConfig(LOGGING_CONFIG)

# Get the loggers
global_logger = logging.getLogger('global_logger')
evaluation_logger = logging.getLogger('evaluation_logger')
