from polyglot.downloader import downloader
print(downloader.supported_languages_table("morph2"))

from polyglot.text import Text, Word
words = ["preprocessing", "processor", "invaluable", "thankful", "crossed"]
for w in words:
      w = Word(w, language="en")
      print("{:<20}{}".format(w, w.morphemes))



import pymorphy2

morph = pymorphy2.MorphAnalyzer()
word = "unhappinessly"
parsed_word = morph.parse(word)[0]
print(parsed_word.normal_form)  # base form
print(parsed_word.tag)