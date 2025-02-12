import logging
import sys

import os.path
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tiktoken
from overrides import override
from sqlalchemy import create_engine, inspect

from base_benchmark import BenchmarkBase, TaskResult

class TrialBenchBenchmark(BenchmarkBase):
    """
    Provides very basic implementation
    """

    def __init__(self, workload_data: dict, halt_on_error=False, intent_based_match=False):
        print("This is comming from the TrialBench implementation")
        super().__init__(workload_data, halt_on_error)
        self.llama_debug = None
        self.token_counter = None
        self.llama_index_query_engine = None
        self.table_names = None
        logging.basicConfig(
            stream=sys.stdout, level=logging.DEBUG
        )  # logging.DEBUG for more verbose output
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    def setup(self, benchmark_config: dict):
        # This takes care of setting up the golden query engine
        # self.engine should contain the golden query engine
        super().setup(benchmark_config)

    def generate_query(self, query_info: dict):
        print("This Generate query function has been hit!")

        device = "auto"
        model_path = "ibm-granite/granite-3.1-2b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # drop device_map if running on CPU
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
        model.eval()
        # change input text as desired
        chat = [
            { "role": "user", "content": "You are a data scientist with expertise in SQL. Your task is to translate the questions in natural language into SQL queries. Here is the inforamtion: "},
        ]
        chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # tokenize the text
        input_tokens = tokenizer(chat, return_tensors="pt").to(device)
        # generate output tokens
        output = model.generate(**input_tokens, 
                                max_new_tokens=100)
        # decode output tokens into text
        output = tokenizer.batch_decode(output)
        # print output
        print(output)

        return output

    @override
    def cleanup(self):
        """
        Cleanup the benchmark. In case you need to do any cleanup.
        """
        super().cleanup()
