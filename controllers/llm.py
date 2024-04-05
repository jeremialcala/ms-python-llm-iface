# -*- coding: utf-8 -*-
import logging
import logging.config

import torch
from inspect import currentframe
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import timeit, configure_logging
from classes import Settings

_set = Settings()
n_gpus = torch.cuda.device_count()
log = logging.getLogger(__name__)
logging.config.dictConfig(configure_logging())


@timeit
def get_llm_elements(use_cuda=False):
    log.info(f"Starting: {currentframe().f_code.co_name}")
    model_name = _set.model_name

    log.info("creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("creating model...")
    if use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={i: _set.max_memory for i in range(n_gpus)}
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    log.info(f"Ending: {currentframe().f_code.co_name}")
    return model, tokenizer

