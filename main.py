# -*- coding: utf-8 -*-
import sys
import logging.config
import keyboard
import torch

import functools
import threading

from inspect import currentframe
from classes import Settings
from utils import configure_logging, timeit
from controllers import get_llm_elements, get_amqp_connection, on_message

_set = Settings()

log = logging.getLogger(__name__)
logging.config.dictConfig(configure_logging())

arg_names = ["--use-cuda", "--queue"]

n_gpus = torch.cuda.device_count()


@timeit
def get_help():
    """
    This get messages from a SQS, process it using a dictionary for common and known concepts
    for the messages not in the database will complete it with transformers using llama models.

    Args:
        --use-cuda: Enables the use of cuda GPU, needs a configuration for pytorch + CUDA.

    """


def process_messages(queue, _use_cuda=False):
    log.info(f"Starting: {currentframe().f_code.co_name}")
    model, tokenizer = get_llm_elements(_use_cuda)
    while True:
        _conn = get_amqp_connection()
        channel = _conn.channel()
        channel.queue_declare(queue=queue, auto_delete=False)
        channel.queue_bind(queue=queue, exchange="chatters", routing_key="key1234")

        # This indicates the number of threads to be used by the message manager
        channel.basic_qos(prefetch_count=1)

        threads = []
        on_message_callback = functools.partial(on_message, args=(_conn, threads, model, tokenizer))
        channel.basic_consume(queue=queue, on_message_callback=on_message_callback)

        channel.start_consuming()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        if keyboard.is_pressed('q'):
            channel.stop_consuming()
            break

    log.info(f"Ending: {currentframe().f_code.co_name}")


if __name__ == '__main__':
    configure_logging()
    log.info(f"Starting: {currentframe().f_code.co_name}")
    if "--help" in sys.argv or len(sys.argv) == 1:
        print(get_help.__doc__)
        sys.exit(0)

    arg_name = [arg for arg in sys.argv if "--" in arg]

    use_cuda = False
    if "--use-cuda" in sys.argv:
        use_cuda = True

    process_messages(_set.queue_name, use_cuda)
    log.info(f"Ending: {currentframe().f_code.co_name}")
    exit(0)
