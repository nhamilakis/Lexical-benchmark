from pathlib import Path

import pandas as pd

from lexical_benchmark import settings
from lexical_benchmark.stats import block_average2_stat as block_stat



class Metric:
    """dataset metric for the given word list"""
    def __init__(self, data:list[str],chunk_size:int=None):

        self.data = data
        self.chunk_size = chunk_size

        if chunk_size:
            # split into chunks
            self.chunk_lst = block_stat.chunk_splitter(data, chunk_size) 
        

    
    def compute_ttr(self)->float:
        """get token/type ratio of the input string"""
        return block_stat.type_token_chunk(self.chunk_lst)

    

    def compute_type_rej_rate(self)->float:
        """get token/type ratio of the input string"""
        
        return block_stat.clean_chunk_list(
                chunks=self.chunk_list, filter_fn=word_clean_fn
            )
        

    def compute_CDI(self,threshold:int,CDI_words:list)->float:
        """get average CDI scores of the given word list"""

        return None

    

