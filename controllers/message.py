# -*- coding: utf-8 -*-
import logging
import logging.config

from classes import Settings
from utils import configure_logging, timeit


_set = Settings()

log = logging.getLogger(__name__)
logging.config.dictConfig(configure_logging())
