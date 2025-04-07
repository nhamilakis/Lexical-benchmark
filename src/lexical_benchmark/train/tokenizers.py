import logging
from pathlib import Path

import datasets

# Set up logging
L = logging.getLogger(__name__)

def load_joined_text(file_path: Path, tokenizer, max_length: int) -> datasets.Dataset:
    """Load text and join utterances to maximize context usage up to max_length."""
    # Read all lines
    L.info(f"Reading text from {file_path}")
    with file_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    L.info(f"Read {len(lines)} lines from file")

    # Tokenize each line separately to get their token counts
    line_tokens = []
    for line in lines:
        tokens = tokenizer.encode(line, add_special_tokens=True)
        line_tokens.append(tokens)

    # Join utterances to form examples that approach max_length
    joined_examples = []
    current_example = []
    current_length = 0

    for tokens in line_tokens:
        # If adding this line would exceed max_length, start a new example
        if current_length + len(tokens) > max_length:
            if current_example:  # Only add if we have something
                joined_examples.append(current_example)
            current_example = tokens
            current_length = len(tokens)
        else:
            # Add to current example
            if current_example:
                # Remove BOS token if not first line to avoid duplicate special tokens
                current_example.extend(tokens[1:])
            else:
                current_example = tokens
            current_length = len(current_example)

    # Add the last example if it exists
    if current_example:
        joined_examples.append(current_example)

    L.info(f"Created {len(joined_examples)} joined examples from {len(lines)} original lines")

    # Convert to features
    features = {"input_ids": [], "attention_mask": []}

    for example in joined_examples:
        # Truncate if somehow still too long
        if len(example) > max_length:
            example = example[:max_length]

        # Create attention mask (all 1s since we have no padding yet)
        attention_mask = [1] * len(example)

        features["input_ids"].append(example)
        features["attention_mask"].append(attention_mask)

    return datasets.Dataset.from_dict(features)


