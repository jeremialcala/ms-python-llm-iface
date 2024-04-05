import logging
import logging.config
from classes import Settings
from random import choice

from functools import wraps
import time
from .logging import configure_logging


log = logging.getLogger(__name__)
logging.config.dictConfig(configure_logging())
settings = Settings()
alpha_numeric = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        print(f'Starting {func.__name__}')
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} completed total time:{total_time:.4f} seconds')
        return result

    return timeit_wrapper

def generate_code(size=32):
    key = ""
    for x in range(0, size):
        key += str(choice(alpha_numeric)).upper()
    return key

