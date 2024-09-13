# Wordlist filtering

We filter words to separate valid from invalid speech, we do it using specific lexicon dictionairies.

All the dictionnairies aggragated give us an approximate of 1,534,810 words. Which gives us a good coverage of the english language.
The ideal would be to use the Google Book's corpus with more that 2M words but it is not open-source (I think?).

### SCOWLv2 - ~676k words

[Aspell](http://wordlist.aspell.net) is a spellchecker created by open-source communities, and
variations of it is used by Mozzila, OpenOffice and various linux distributions.

Their site has various dictionairies & tools to make spellcheckers.

We use their tool in this page to generate a wordlist => [Create wordlist](http://app.aspell.net/create)

The following parameters are used :

- diacritic=strip (replaces words like café with cafe)
- max_size=95 (this is the biggest available has **~676k words**)
- max_variant=3 (this is the biggest available allows multiple variants of the same word)
- special=hacker (includes computer science related vocabulairy, we excluded roman numerals)
- spelling: US, GB( s & z variants)


**CITATION**: None copyright notice in github gives various thanks to open-sourced projects that contributed to lexicon.


### Kaiki.org - ~1M words

Kaikki.org is a digital archive and a data mining group. We aim to make our digital heritage more 
accessible and useful for people, researchers, linguists, software developers, and artificial 
intelligence (AI).

We used the All-English forms which is their biggest dictionairy for english available you can find it [here](https://kaikki.org/dictionary/English/index.html).

We did some post-processing to extract only the words from the dictionairy.


**CITATION**:  If you use this data in academic research, please cite [Tatu Ylonen: Wiktextract: Wiktionary as Machine-Readable Structured Data](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.140.pdf), Proceedings of the 13th Conference on Language Resources and Evaluation (LREC), pp. 1317-1325, Marseille, 20-25 June 2022. Linking to the relevant page(s) under https://kaikki.org would also be greatly appreciated.


### YAWL - ~264k words


### Childes extras

Using the tags from CHILDES we create a lexicon of non-words