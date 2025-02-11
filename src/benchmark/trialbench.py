import logging
import sys

import os.path
import time
import traceback

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

        print("This is comming from the TrialBench implementation")

    @override
    def cleanup(self):
        """
        Cleanup the benchmark. In case you need to do any cleanup.
        """
        super().cleanup()
