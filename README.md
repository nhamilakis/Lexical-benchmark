# Lexical-benchmark
The lexical benchmark  project provides more finegrained human-model comparison on lexical level by carefully matching the test word set and evalauation tasks. 

It covers the following two aspects: i) receptive vocabulary  ii) expressive vocabualry


util.py: common functions for the scripts below

Human.py: get freq of CDI receptive/expressive vocab from CHILDES parents'/children's utterances
		input: 1.CHILDES transcript; 2.CDI data
        	output: all are in the same folder named as lang_condition 
        		1. all the selected transcripts by interlocutor in CHILDES
        		2. the freq list of all the words in the selected transcripts 
        		3. the freq list of all the words in the selected transcripts with the overlapping of CDI data

Link to the CHILDES dataset for threshold estimation: https://drive.google.com/file/d/1cGO8j0nP2J_vViaQeUjdZHTFkZfZ6J_s/view?usp=sharing
Link to the AO-CHILDES: https://github.com/UIUCLearningLanguageLab/AOCHILDES


Recep_model:
	utils.py: common functions used in compute_prob.py

	compute_prob.py: compute probability of the word-non-word pairs: for both acoustic and textual inputs
		input: 1. model checkpoints; 2.word-non-word pairs to be evaluated 
		output: text file containing probabilities

	generate_scores.py: calculate the lexical scores based on the results of wuggy test
		input: The text file containing the word probabilities 
		output: .csv file containing test scores

	


Exp_model:	
	Reference_model.py: train the reference model to optimize hyperparameters of model generations
		input: Training dataset and months to be simulated
		output: trained model checkpoints
	Generation.py: LSTM model generation with/out prompts using beam search; sampling method(random, topk, topp)


