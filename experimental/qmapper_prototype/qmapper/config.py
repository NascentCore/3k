import logging
import logging.config

def setup_logging(default_path='logging.yaml', default_level=logging.INFO):
    """Setup logging configuration"""
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })

setup_logging()

qmapper_logger = logging.getLogger(__name__)
qmapper_logger.info('Qmapper Imported')