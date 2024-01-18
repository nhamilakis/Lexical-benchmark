# Machine CDI: Lexical-benchmark
This repository focuses on a detailed comparison between human and model performance at the lexical level. The project comprises two main components: Receptive Vocabulary (Recep_model) and Expressive Vocabulary (Exp_model).

Receptive Vocabulary (Recep_model)
Util.py
Common functions for the scripts.

Human.py
Retrieves CDI receptive/expressive vocabulary frequencies from CHILDES parents' and children's utterances.

CHILDES Dataset: Download

AO-CHILDES Repository: Link

Expressive Vocabulary (Exp_model)
Recep_model/utils.py
Common functions for compute_prob.py.

compute_prob.py
Computes word-non-word pair probabilities for acoustic and textual inputs.

generate_scores.py
Calculates lexical scores based on Wuggy test results.

Exp_model
Reference_model.py
Trains the reference model to optimize hyperparameters.

Generation.py
LSTM model generation with/without prompts using beam search and sampling methods (random, topk, topp).


