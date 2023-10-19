# Lexical-benchmark
This project provides more finegrained human-model comparison on lexical level. This project is divided into following components: i) model training;  ii) model evaluation;   iii) plot the data


i)  Training


ii) Eval
util.py: common functions for the scripts below

Recep_human.py: get freq of CDI receptive vocab from CHILDES parents' utterances
		input: 1.CHILDES transcript; 2.CDI data
        	output: all are in the same folder named as lang_condition 
        		1. all the selected transcripts by interlocutor in CHILDES
        		2. the freq list of all the words in the selected transcripts 
        		3. the freq list of all the words in the selected transcripts with the overlapping of CDI data

Exp_human.py: get freq of CDI receptive vocab from CHILDES children's utterances and also fitness between accumulator and CDI data
		input: 1.CHILDES transcript; 2.CDI data
		output: 1.normalized words' frequency in CDI, based on children's utterances in CHILDES; 2. figures of accumulator and CDI


Recep_model:
	utils.py: common functions used in compute_prob.py

	compute_prob.py: compute probability of the word-non-word pairs: for both acoustic and textual inputs
		input: 1. model checkpoints; 2.word-non-word pairs to be evaluated 
		output: text file containing probabilities

	generate_scores.py: calculate the lexical scores based on the results of wuggy test
		input: The text file containing the word probabilities 
		output: .csv file containing test scores

	TO DO: write the pipeline python file fo rthe whole procedure


Prod:	
	Reference_model.py: train the reference model to optimize hyperparameters of model generations
		input: Training dataset and months to be simulated
		output: trained model checkpoints



