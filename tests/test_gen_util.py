from pathlib import Path
import argparse
import logging
import string
import torch
import random
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from lexical_benchmark.utils import hf_util 
from lexical_benchmark.utils import gen_util




class Logger:
    """Utility class for logging configuration"""
    
    @staticmethod
    def setup(output_dir: str, filename: str = "inference.log") -> logging.Logger:
        """Configure and return a logger."""
        log_path = Path(output_dir) / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(str(log_path)),
                logging.StreamHandler()
            ],
            force=True
        )
        return logging.getLogger(__name__)


class TextGenerator:
    """Handles text generation using transformer models."""
    
    def __init__(self, model_path: str,
                 model_max_length: int = 1024,
                 model_type: str = "transformer",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to the model
            chars: String containing all characters for tokenization
            model_max_length: Maximum sequence length
            model_type: Type of model ("transformer" or "lstm")
            device: Device to run on
        """
        # Initialize tokenizer first
        chars = string.ascii_letters
        self.tokenizer = hf_util.CharacterTokenizer(chars=chars, model_max_length=model_max_length)
        self.model = self._load_model(model_path, model_type, device)
        self.device = device
        self.model.eval()
        self.max_length = min(model_max_length, getattr(self.model.config, 'n_positions', model_max_length))
    
    def _load_model(self, model_path: str, model_type: str, device: str) -> PreTrainedModel:
        """Load the model from path."""
        try:
            if model_type.lower() == "lstm":
                config = gen_util.LSTMConfig.from_pretrained(model_path)
                model = gen_util.LSTMForLanguageModeling.from_pretrained(model_path, config=config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            print('Model has been loaded')
            return model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    

    #TODO: merge it with hf_util
    def add_special_tokens(self, special_token_lst: list[str]=["'", "|"]):
        # add the new token for abbreviation
        for special_token in special_token_lst:
            self.tokenizer.add_tokens(special_token)
            print("Added special tokens to the tokenizer")
        

    
    def generate_text(self, word_num: int, temp_lst: List[float]) -> Dict[str, str]:
        """Generate text with different temperatures."""
        results = {}
        # Use a valid token ID from the tokenizer's vocabulary
        random_token_id = random.randint(0, 25)

        for temp in temp_lst:
            with torch.no_grad():
                input_ids = torch.tensor([[random_token_id]], device=self.device)
                gen = self.tokenizer.decode([random_token_id])
                bar_count = 0
                
                while bar_count < word_num and input_ids.shape[1] < self.max_length:
                    curr_length = input_ids.shape[1]
                    position_ids = torch.arange(curr_length, device=self.device).unsqueeze(0)
                    
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        #max_new_tokens=1,
                        max_length=curr_length + 1,
                        do_sample=True,
                        temperature=temp,
                        num_beams=1,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        position_ids=position_ids,
                        use_cache=True
                    )
                    
                    new_token = outputs[0, -1].item()
                    decoded_token = self.tokenizer.decode([new_token])
                    
                    if decoded_token == "|":
                        bar_count += 1
                    
                    gen += decoded_token
                    input_ids = outputs
                    
                    if input_ids.shape[1] >= self.max_length - 2:
                        input_ids = input_ids[:, :1]
                
                results[f"unprompted_{temp}"] = gen
                
        return results






class BatchProcessor:
    """Handles batch processing of text generation."""
    
    def __init__(self, generator: TextGenerator, save_path: Path, chunk_size: int = 10):
        """Initialize batch processor."""
        self.generator = generator
        self.save_path = Path(save_path)
        self.chunk_size = chunk_size
        self.logger = Logger.setup(self.save_path)
    
    def process_batch(self, batch: pd.DataFrame, temp_lst: List[float]) -> pd.DataFrame:
        """Process a single batch of data."""
        temp_columns = [f"unprompted_{temp}" for temp in temp_lst]
        results = []
        
        for _, row in batch.iterrows():
            try:
                result = self.generator.generate_text(row['sent_len'], temp_lst)
                results.append(pd.Series(result))
            except Exception as e:
                self.logger.error(f"Error in generation: {str(e)}")
                results.append(pd.Series({col: "" for col in temp_columns}))
            finally:
                torch.cuda.empty_cache()
                
        batch[temp_columns] = pd.DataFrame(results, index=batch.index)
        return batch
    
    def process_dataframe(self, df: pd.DataFrame, temp_lst: List[float], 
                         save_interval: int, resume: bool = False) -> pd.DataFrame:
        """Process entire dataframe with save intervals."""
        gen = pd.DataFrame()
        resume_file = self.save_path / "gen_intermediate.csv"
        
        if resume and resume_file.is_file():
            gen = pd.read_csv(resume_file).loc[:, "month":]
            df = df.iloc[gen.shape[0]:]
            self.logger.info(f"Resuming generation from checkpoint. Rows processed: {len(gen)}")
        
        total_rows = len(df)
        chunks = [df.iloc[i:i + save_interval] for i in range(0, total_rows, save_interval)]
        
        for chunk in tqdm(chunks, desc="Processing chunks"):
            processed_chunks = []
            
            for i in range(0, len(chunk), self.chunk_size):
                batch = chunk.iloc[i:i + self.chunk_size].copy()
                processed_batch = self.process_batch(batch, temp_lst)
                processed_chunks.append(processed_batch)
                torch.cuda.empty_cache()
            
            processed_df = pd.concat(processed_chunks)
            gen = pd.concat([gen, processed_df])
            
            # Save intermediate results
            gen.to_csv(self.save_path / "gen_intermediate.csv")
            self.logger.info(f"Saved intermediate results. Total rows processed: {len(gen)}")
            
        return gen

