# Build dataset/pre-process

This folder contains methods & recipes to create the dataset structure 
from their source.

Data created is in its raw format (without any cleaning).
To clean-up data you need to use the dataloaders/preprocess, in combination
with text_lib.


## Contents

#### Childes

The goal of the CHILDES pre-processing is to extract data from the `.cha` format, 
nativelly used in the CHILDES dataset and convert it into a `txt` format.

We export 2 different types of `txt` formats:

**by_type**: In this format speech is separated into child & adult speech.

LANG/by_type/adult/*.raw
LANG/by_type/adult/*.raw

We only keep the transcription part of the `.cha` format, and remove the speaker_id.
Text is in raw format as txt files with one sentence per line.


**by_dialog**: In this format we keep the dialog information of the childes. 

LANG/by_dialog/*.raw.json

The JSON files contain a list of dialog items which are a tuple of : (SPEAKER_ID, TXT)


#### STELA

The goal of STELA preprocessing is to re-format the InfTrain dataset to be a transcription based dataset.

To accomplish this we use the transcriptions of the audio-books and recreate the original splits of : 50h, 100h, 200h, 400h, 800h, 1600h, 3200h

Information on how to match books to split can be found in csv:

-  `InfTrain/metadata/matched2.csv`

The result format of the preprocessed version is : 

LANG/by_hours/
    50h/**/
        books/[BOOK_ID].raw
    ...

LANG/by_genre/
    [GENRE_NAME]/
        [BOOK_ID].raw



#### ChildRealistic