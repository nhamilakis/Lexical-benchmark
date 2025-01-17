"""Generate word-like units based on the target distr final results should be model by model."""

import logging
import random
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from lexical_benchmark.utils import gen_util


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(str(Path(output_dir) / "inference.log")), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_model(
    model_path: str, model_type: str = "LSTM", device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> transformers.PreTrainedModel:
    """Load the trained model."""
    try:
        if model_type.lower() == "lstm":
            config = gen_util.LSTMConfig.from_pretrained(model_path)
            model = gen_util.LSTMForLanguageModeling.from_pretrained(model_path, config=config)
        else:
            # Load default transformer model
            model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model
    except Exception as e: # TODO(@Jing): make the exception more specific
        raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e


def generate(word_num, tokenizer, model, device, temp_lst):
    """Transformer model generation.

    Generate sequences for a given number of `|` tokens for each temperature in `temp_lst`.
    Returns a dictionary with column names as keys and generated sequences as values.
    """
    results = {}
    # only generate from the lower-cased characters
    random_token_id = random.randint(0, 25)

    for temp in temp_lst:
        bar_count = 0
        prev_bar = False

        # Decode the initial token
        input_ids = torch.tensor([[random_token_id]]).to(device)
        gen = tokenizer.decode(random_token_id)

        # Start generating tokens iteratively
        while bar_count < word_num:
            # Generate the next token(s)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=1024,#input_ids.shape[1] + 1, # issue with Increment by 1 token
                num_beams=1,
                num_return_sequences=1,
                temperature=temp,
                top_k=0,
                top_p=1,
                do_sample=True,
                early_stopping=False,
            )

            # Get the newly generated token
            new_token = outputs[0, -1].item()  # Last token in the generated sequence
            decoded_token = tokenizer.decode(new_token)

            # Check if the token is `|` and avoid consecutive `|` tokens
            if decoded_token == "|":
                if not prev_bar:
                    bar_count += 1
                    prev_bar = True  # Mark that we've generated a `|`
            else:
                prev_bar = False  # Reset if it's not a `|`

            # Add the decoded token to the generated sequence
            gen += decoded_token
            # Update input_ids for the next token generation
            input_ids = torch.cat((input_ids, outputs[0, -1:].unsqueeze(0)), dim=1)

        # Add the result to the dictionary with the appropriate column name
        column_name = f"unprompted_{temp}"
        results[column_name] = gen

    return results
