# Download the corpora

We match the machine CDI test sets with human CDI words in frequency, assuming the same amount of exposure. The human CDI words frequency is derived from [American-English child-directed speech transcripts (AO-CHILDES)](https://github.com/UIUCLearningLanguageLab/AOCHILDES) and the machine CDI word from the training set of langauge model(audiobook orthographic transcripts). We also use CELEX corpus as frequency reference considering the potential match with these two corpora: 

1) register of the discourse (AO-CHILDES and CELEX are both informal conversations); and 
2) age of the speaker (Audiobook transcripts and CELEX are both produced by adults)


The corpora files are available here: 

1) Human CDI: [Wordbank](scripts/select_test/stat/corpus/human)
2) Machine CDI: [CELEX](scripts/select_test/stat/corpus/SUBTLEX_US.xls), [audiobook](https://drive.google.com/file/d/1Ec9vQHVUWs2t6pW1LxhCNrzRTGQnoWRV/view?usp=sharing), [AO-CHILDES](https://github.com/UIUCLearningLanguageLab/AOCHILDES)


Alternatively, you can click on the following links to download the test sets: 

1) Human CDI: [Wordbank](https://wordbank.stanford.edu/data?name=item_data)
2) Machine CDI: [CELEX](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus4.zip)



# Principles of frequency match

You can apply three types of match between human and machine CDI words: 

1) match by range: match the overall frequency range;
2) match by bin range: match each bin's frequency range;
3) match by count: match each bin's word count distribution;






