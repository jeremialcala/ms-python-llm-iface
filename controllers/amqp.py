import pika
import threading
import logging.config
import functools
from inspect import currentframe
from classes import Settings
from utils import configure_logging, timeit

_set = Settings()
log = logging.getLogger(__name__)
logging.config.dictConfig(configure_logging())


@timeit
def get_amqp_connection():
    log.info(f"Starting: {currentframe().f_code.co_name}")
    credentials = pika.credentials.PlainCredentials(username=_set.qms_user, password=_set.qms_password)
    conn_parameters = pika.ConnectionParameters(host=_set.qms_server, credentials=credentials)
    log.info(f"Ending: {currentframe().f_code.co_name}")
    return pika.BlockingConnection(conn_parameters)


@timeit
def on_message(_channel, method_frame, header_frame, body, args):
    log.info(f"Starting: {currentframe().f_code.co_name}")
    (_connection, _threads, model, tokenizer) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(_connection, _channel, delivery_tag, body))
    t.start()
    _threads.append(t)
    log.info(f"Ending: {currentframe().f_code.co_name}")


@timeit
def do_work(connection, channel, delivery_tag, body):
    thread_id = threading.get_ident()
    fmt1 = 'Thread id: {} Delivery tag: {} Message body: {}'
    log.info(fmt1.format(thread_id, delivery_tag, body))
    # TODO: Send the message in the body to LLM
    cb = functools.partial(ack_message, channel, delivery_tag)
    connection.add_callback_threadsafe(cb)


@timeit
def ack_message(channel, delivery_tag):
    """Note that `channel` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if channel.is_open:
        channel.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass
