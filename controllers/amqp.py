import json
import pika
import threading
import logging.config
import functools

from inspect import currentframe
from classes import Settings
from utils import configure_logging, timeit
from .llama_generator import Llama

from typing import List, Literal, Optional, Tuple, TypedDict

_set = Settings()
log = logging.getLogger(__name__)
logging.config.dictConfig(configure_logging())

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

# If a question does not make any sense,
# or is not factually coherent, explain why instead of answering something not correct.
# If you don't know the answer to a question, please don't share false information.


DEFAULT_SYSTEM_PROMPT = ("Tu nombre es VotaBot, tu rol es de asistente electoral: "
                         "  1. Entrega solo respuestas en español, "
                         "  2. Se amable y cordial al responder. "
                         "  3. Solo entrega información sobre como funcionan los procesos electorales en Venezuela."
                         "  4. Si la pregunta no hace sentido o no es coherente pide mas informacion."
                         "  5. Si no sabes la respuesta a la pregunta, no compartas informacion falsa o sin verificar.")

Dialog = List[Message]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


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
    t = threading.Thread(target=do_work, args=(_connection, _channel, delivery_tag, body, model, tokenizer))
    t.start()
    _threads.append(t)
    log.info(f"Ending: {currentframe().f_code.co_name}")


@timeit
def do_work(connection, channel, delivery_tag, body, model, tokenizer):
    thread_id = threading.get_ident()
    fmt1 = 'Thread id: {} Delivery tag: {} Message body: {}'
    log.info(fmt1.format(thread_id, delivery_tag, body))
    _body = json.loads(body.decode("utf-8"))
    # TODO: Send the message in the body to LLM
    try:
        match _body["jwe_body"]["type"]:

            case "page":
                text = _body['jwe_body']['message']['message']['text']
                log.info(text)

                msg = Message(role="system", content=DEFAULT_SYSTEM_PROMPT)

                dialogs = [[msg, Message(role="user", content=text)]]

                # [log.info(answer) for answer in chat_completion(model, tokenizer, dialogs)]

                g_text = generate_answer(
                    dialog=text,
                    _model=model,
                    _tokenizer=tokenizer,
                    use_cuda=True)

                log.info(g_text)
                # [log.info(txt) for txt in g_text]

            case _:
                pass

        cb = functools.partial(ack_message, channel, delivery_tag)
        connection.add_callback_threadsafe(cb)

    except Exception as e:
        log.error(e.__str__())
        pass


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


@timeit
def generate_answer(dialog, _model, _tokenizer, use_cuda: bool = True, max_length=1024):
    print(f"we are using cuda: <---> {use_cuda}")
    if use_cuda:
        t_prompt = _tokenizer(
            f"{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {dialog} {E_INST}",
            return_tensors="pt").input_ids.cuda()
    else:
        t_prompt = _tokenizer(
            f"{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {dialog} {E_INST}",
            return_tensors="pt").input_ids

    generated = _model.generate(
        t_prompt,
        max_length=max_length,
        # max_new_tokens=350,
    )[0]

    return _tokenizer.decode(generated)
