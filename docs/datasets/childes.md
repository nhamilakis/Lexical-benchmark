# CHILDES DATASET

The CHILDES dataset is a data can be found in the [childes.talkbank.org](https://childes.talkbank.org) website.

We used the North-American & UK English subsets for this study.

To download the data you can use : 

```bash
wget https://childes.talkbank.org/access/Eng-NA/0-Eng-NA-MOR.zip
wget https://childes.talkbank.org/access/Eng-UK/0-Eng-UK-MOR.zip
```

## Cleanup

For the cleanup of the CHILDES Datasets ....TBA


We used the Syntax of CHILDES annotation and came up with a list of choice rules on how to process tagging/punctuation/metadata etc...
You can find the url to the full documentation of the CHAT format [annotations here](https://talkbank.org/manuals/CHAT.html).

<details>
<summary>Rules & Choices</summary>
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">TAG</th>
    <th class="tg-0pky">Name</th>
    <th class="tg-0pky">Example</th>
    <th class="tg-0pky">NB Occurences</th>
    <th class="tg-0pky">CHOICE</th>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">Extra</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">"Adult" Speech</td>
    <td class="tg-0pky">Key Child Speech</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">"@o"</td>
    <td class="tg-0pky">onomatopoeia</td>
    <td class="tg-0pky">woofwoof@o</td>
    <td class="tg-0pky">45394</td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@p</td>
    <td class="tg-0pky">Phonological consistent forms</td>
    <td class="tg-0pky">aga@p</td>
    <td class="tg-0pky">23193</td>
    <td class="tg-0pky">DISCARD (Should be 0)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">PCFs are early forms that are phonologically consistent, but whose meaning is unclear to the transcriber. Often these forms are protomorphemes.</td>
  </tr>
  <tr>
    <td class="tg-0pky">@b</td>
    <td class="tg-0pky">Babbling</td>
    <td class="tg-0pky">abame@b</td>
    <td class="tg-0pky">3230</td>
    <td class="tg-0pky">DISCARD (Should be 0)</td>
    <td class="tg-0pky">keep as non-word</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@wp</td>
    <td class="tg-0pky">word play (simillar to babbling)</td>
    <td class="tg-0pky">goobarumba@wp</td>
    <td class="tg-0pky">4704</td>
    <td class="tg-0pky">DISCARD (Should be 0)</td>
    <td class="tg-0pky">keep as non-word</td>
    <td class="tg-0pky">In older children produces forms that may sound much like the forms of babbling, but which arise from a slightly different process. It is best to use the @b for forms produced by children younger than 2;0 and @wp for older children.</td>
  </tr>
  <tr>
    <td class="tg-0pky">@c</td>
    <td class="tg-0pky">Child Invented Form</td>
    <td class="tg-0pky">gumma@c (meaning sticky)</td>
    <td class="tg-0pky">20063</td>
    <td class="tg-0pky">DISCARD (Should be 0)</td>
    <td class="tg-0pky">keep as non-word</td>
    <td class="tg-0pky">Like babbling but with an understandable meaning</td>
  </tr>
  <tr>
    <td class="tg-0pky">@f</td>
    <td class="tg-0pky">Family Specific Form</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">keep as non-word</td>
    <td class="tg-0pky">A Child Invented Form that is used by the whole family</td>
  </tr>
  <tr>
    <td class="tg-0pky">@d</td>
    <td class="tg-0pky">Dialect Word</td>
    <td class="tg-0pky">younz@d (meaning you)</td>
    <td class="tg-0pky">4768</td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@s</td>
    <td class="tg-0pky">Second (or other) Language</td>
    <td class="tg-0pky">perro@s:es (meaning dog in spannish)</td>
    <td class="tg-0pky">3049</td>
    <td class="tg-0pky">DISCARD</td>
    <td class="tg-0pky">DISCARD</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@n</td>
    <td class="tg-0pky">Neologism</td>
    <td class="tg-0pky">breaked@n (meaning broke)</td>
    <td class="tg-0pky">2088</td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@si</td>
    <td class="tg-0pky">Singing</td>
    <td class="tg-0pky">lalala@si</td>
    <td class="tg-0pky">2449</td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@i</td>
    <td class="tg-0pky">interjection/interaction</td>
    <td class="tg-0pky">uhhuh@i<br></td>
    <td class="tg-0pky">839</td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">Interjections can be indicated in standard ways, making the use of the @i notation usually not necessary. Instead of transcribing “ahem@i,” one can simply transcribe ahem following the conventions listed later.</td>
  </tr>
  <tr>
    <td class="tg-0pky">@t</td>
    <td class="tg-0pky">Test Word (Various words spoken by Investigator)</td>
    <td class="tg-0pky">monkey@i</td>
    <td class="tg-0pky">508</td>
    <td class="tg-0pky">KEEP </td>
    <td class="tg-0pky">keep</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@q</td>
    <td class="tg-0pky">Meta-Linguistic form</td>
    <td class="tg-0pky">no if@q-s or but@q-s</td>
    <td class="tg-0pky">5634</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">can be used to either cite or quote single standard words or special child forms</td>
  </tr>
  <tr>
    <td class="tg-0pky">@u</td>
    <td class="tg-0pky">Phonetic Transcription (in unibet)</td>
    <td class="tg-0pky">den@u, Ef@u, pUlEf@u, krElo@u, sumerti@u, A@u</td>
    <td class="tg-0pky">714</td>
    <td class="tg-0pky">KEEP (MAYBE ADD TO DICTIONARY)</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@l</td>
    <td class="tg-0pky">Letter</td>
    <td class="tg-0pky">M@l i@l k@l e@l (Spelling of the word Mike)<br><br>ten o'clock a@l m@l” for 10:00 AM,</td>
    <td class="tg-0pky">40297</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@k</td>
    <td class="tg-0pky">Multiple letters (simillar to @l)</td>
    <td class="tg-0pky">ka@k (Japanese “ka”)</td>
    <td class="tg-0pky">232</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">@z</td>
    <td class="tg-0pky">Custom codes</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">DEPENDS ON USAGE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">@z:sc</td>
    <td class="tg-0pky">Check Eng-NA/Braunwald/030009</td>
    <td class="tg-0pky">328</td>
    <td class="tg-0pky">KEEP (Remove tag)</td>
    <td class="tg-0pky">KEEP (Remove tag)</td>
    <td class="tg-0pky">Only used in Eng-NA/Braunwald with the specifier ("@z:sc"), i am not sure the usage (could not find explanation) but the words seem unrelated, and might be just relative to the study. </td>
  </tr>
  <tr>
    <td class="tg-0pky">@x</td>
    <td class="tg-0pky">Excluded Words</td>
    <td class="tg-0pky">Check: Eng-UK/OdiaMAIN/C12_1108 , Eng-UK/OdiaMAIN/C10_1010<br>Check: Eng-NA/NewmanRatner/Interviews/24/6493TM</td>
    <td class="tg-0pky">64</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">No clear pattern, check plot of words excluded (&amp; refs)</td>
  </tr>
  <tr>
    <td class="tg-0pky">@g</td>
    <td class="tg-0pky">General Special Form</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky">was not present in audio</td>
  </tr>
  <tr>
    <td class="tg-0pky">@m</td>
    <td class="tg-0pky">Not a tag</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">KEEP</td>
    <td class="tg-0pky">KEEP (Remove tag)</td>
    <td class="tg-0pky">Propably an error, word looks like babbling</td>
  </tr>
  <tr>
    <td class="tg-0pky">TAG</td>
    <td class="tg-0pky">Name</td>
    <td class="tg-0pky">Example</td>
    <td class="tg-0pky">NB Occurences</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">CHOICE</td>
    <td class="tg-0pky">Extra</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">"&amp;" TAGS</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">TAG</td>
    <td class="tg-0pky">Name</td>
    <td class="tg-0pky">Example</td>
    <td class="tg-0pky">NB Occurences</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">CHOICE</td>
    <td class="tg-0pky">Extra</td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;+</td>
    <td class="tg-0pky">Phonological Fragments</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">1652</td>
    <td class="tg-0pky">DISCARD</td>
    <td class="tg-0pky">DISCARD</td>
    <td class="tg-0pky">Sometimes words, sometimes sounds, sometimes letters</td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;-</td>
    <td class="tg-0pky">Fillers</td>
    <td class="tg-0pky">&amp;-uhm</td>
    <td class="tg-0pky">33933</td>
    <td class="tg-0pky">KEEP (ADD TO dictionary &amp; Normalise) </td>
    <td class="tg-0pky">Keep as non-words</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;~</td>
    <td class="tg-0pky">NonWords</td>
    <td class="tg-0pky">&amp;~stati, &amp;~boun, &amp;~shor</td>
    <td class="tg-0pky">39579</td>
    <td class="tg-0pky">KEEP (ADD TO dictionary &amp; Normalise) </td>
    <td class="tg-0pky">KEEP as non-words</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;=0</td>
    <td class="tg-0pky">Ommitted word</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">$=ACTION:SUB</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">&amp;=cries<br>&amp;=laughs<br>&amp;=imit:lion<br>&amp;=moves:doll<br></td>
    <td class="tg-0pky">21070</td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;*</td>
    <td class="tg-0pky">Interposed Word</td>
    <td class="tg-0pky"><span style="font-style:italic">*CHI:  when I was over at my friend’s house &amp;*</span>MOT:mhm the dog tried to lick me all over.</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;{l=*</td>
    <td class="tg-0pky">Long events</td>
    <td class="tg-0pky">&amp;{l=laughs    are you serious about that     &amp;}l=laughs</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">&amp;{n=* </td>
    <td class="tg-0pky">Long Nonvocal Event</td>
    <td class="tg-0pky">&amp;}n=waving:hands bye mom &amp;}n=waving:hands</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">Other forms</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">text(text)text</td>
    <td class="tg-0pky">Noncompletion of a Word</td>
    <td class="tg-0pky">I been sit(ting) all day </td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">DISCARD PARENTHESIS</td>
    <td class="tg-0pky">DISCARD PARENTHESIS</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">text_text_text</td>
    <td class="tg-0pky">Compounds and Linkages</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">SPLIT WORDS</td>
    <td class="tg-0pky">Split into multiple words</td>
    <td class="tg-0pky">Compound a collection of words into one linguistic entintiy</td>
  </tr>
  <tr>
    <td class="tg-0pky">A_B_C</td>
    <td class="tg-0pky">Accronyms</td>
    <td class="tg-0pky">FBI as F_B_I<br>m_and_m-s for the M&amp;M candy.</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">SPLIT &amp; KEEP</td>
    <td class="tg-0pky">SPLIT &amp; KEEP</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">‡</td>
    <td class="tg-0pky">Satellite Marker</td>
    <td class="tg-0pky">no ‡ Mommy no go</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">.?!</td>
    <td class="tg-0pky">End of phrase punctuation</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">,;:</td>
    <td class="tg-0pky">Separators</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">↑↓</td>
    <td class="tg-0pky">Tone Direction</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">ˈ</td>
    <td class="tg-0pky">Primary Stress</td>
    <td class="tg-0pky">baby want baˈna:nas ?</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Merge Word</td>
    <td class="tg-0pky">(UNICODE: U02C8 &amp; U02CC)</td>
  </tr>
  <tr>
    <td class="tg-0pky">:</td>
    <td class="tg-0pky">Secondary Stress</td>
    <td class="tg-0pky">baby want baˈna:nas ?</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Merge Word</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">^</td>
    <td class="tg-0pky">Pause Between Syllables</td>
    <td class="tg-0pky">is that a rhi^noceros ?</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">MERGE</td>
    <td class="tg-0pky">MERGE?</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">≠</td>
    <td class="tg-0pky">Blocking</td>
    <td class="tg-0pky">≠Thing</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[^ text]</td>
    <td class="tg-0pky">Complex Local Event</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">(.) (..)</td>
    <td class="tg-0pky">Pause</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">(XXX) (XX:XX)</td>
    <td class="tg-0pky">Long pause</td>
    <td class="tg-0pky">I don't (0.15) know .<br>I don't (1:05.15) know .</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+...</td>
    <td class="tg-0pky">Trailing off</td>
    <td class="tg-0pky">smells good enough for +...</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+..? </td>
    <td class="tg-0pky">Trailing Off of a Question</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+!?</td>
    <td class="tg-0pky">Question With Exclamation</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+/.</td>
    <td class="tg-0pky">Interruption</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+/? </td>
    <td class="tg-0pky">Interruption of a Question  </td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+//. </td>
    <td class="tg-0pky">Self-Interruption   </td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+//? </td>
    <td class="tg-0pky">Self-Interrupted Question</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+.</td>
    <td class="tg-0pky">Transcription Break</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">“</td>
    <td class="tg-0pky">Quotation</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky">Unicode 201C 201D</td>
  </tr>
  <tr>
    <td class="tg-0pky">+"/. </td>
    <td class="tg-0pky">Quotation Follows</td>
    <td class="tg-0pky">*CHI:     +" please give me all of your honey.<br>*CHI:     the little bear said +".</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+". </td>
    <td class="tg-0pky">Quotation Precedes</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+" </td>
    <td class="tg-0pky">Quoted Utterance</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+^</td>
    <td class="tg-0pky">Quick Uptake</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+, </td>
    <td class="tg-0pky">Self Completion</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">++</td>
    <td class="tg-0pky">Other Completion</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">&lt;text&gt; </td>
    <td class="tg-0pky">Scoped Symbols</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">KEEP TEXT REMOVE TAGS</td>
    <td class="tg-0pky">Allows to group a phrase and annotate it, see Stressing</td>
  </tr>
  <tr>
    <td class="tg-0pky"> ·0_1073· </td>
    <td class="tg-0pky">Time Alignment</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[=! text]</td>
    <td class="tg-0pky">Paralinguistic Material</td>
    <td class="tg-0pky">that's mine [=! cries].</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[!] </td>
    <td class="tg-0pky">Stressing</td>
    <td class="tg-0pky">Billy, would you please &lt;take your shoes off&gt; [!].</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[!!]</td>
    <td class="tg-0pky">Contrastive Stressing</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[# time]</td>
    <td class="tg-0pky">Duration</td>
    <td class="tg-0pky">I could use &lt;all of them&gt; [# 2.2] for the party.</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[= text] </td>
    <td class="tg-0pky">Explanation</td>
    <td class="tg-0pky">don't look in there [= closet]!</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[: text]  </td>
    <td class="tg-0pky">Replacement</td>
    <td class="tg-0pky">whyncha [: why don’t you] just be quiet!</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[:: text] </td>
    <td class="tg-0pky">Replacement of Real Word</td>
    <td class="tg-0pky">piece [:: peach] </td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[=? text] </td>
    <td class="tg-0pky">Alternative Transcription</td>
    <td class="tg-0pky">we want &lt;one or two&gt; [=? one too].</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[% text]</td>
    <td class="tg-0pky">Comment on Main Line</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[?]</td>
    <td class="tg-0pky">Best Guess</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[*]</td>
    <td class="tg-0pky">Error Marking</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[&gt;] </td>
    <td class="tg-0pky">Overlap Precedes</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">+&lt;</td>
    <td class="tg-0pky">Lazy Overlap</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[/] </td>
    <td class="tg-0pky">Repetition</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[x N]</td>
    <td class="tg-0pky">Multiple Repetitions</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[//] </td>
    <td class="tg-0pky">Retracing</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[///] </td>
    <td class="tg-0pky">Reformulation</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[/-] </td>
    <td class="tg-0pky">False Start Without Retracing</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[/?] </td>
    <td class="tg-0pky">Unclear Retracing Type</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">[^c] </td>
    <td class="tg-0pky">Clause Delimiter</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky"> [- text] </td>
    <td class="tg-0pky">Language Precodes (Multilingual)</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Does not apply</td>
    <td class="tg-0pky">Language code in language switching</td>
  </tr>
  <tr>
    <td class="tg-0pky">[+ text] </td>
    <td class="tg-0pky">Postcodes</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">REMOVE</td>
    <td class="tg-0pky">Metalinguistic information added at the end of the phrase dependes on the study, no list given.</td>
  </tr>
  <tr>
    <td class="tg-0pky">%pic file.jpg</td>
    <td class="tg-0pky">Extra picture for context</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">%txt: cat.txt</td>
    <td class="tg-0pky">Extra text for context</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky"></td>
  </tr>
</tbody></table>
</details>