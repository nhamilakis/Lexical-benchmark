import logging

from lexical_benchmark import settings

from .extract import extract_from_cha

logging.basicConfig(level=logging.DEBUG)
logging.debug("What is going on ?")

_file = settings.PATH.code_root / "data/childes_sample/Bates/Free20/jane.cha"
_data = extract_from_cha(_file)
print(_data)
