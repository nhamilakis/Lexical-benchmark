import typing as t
from pathlib import Path

from lexical_benchmark.utils import deprecated


class Testing:
    """WTF."""

    @property
    def malakies(self) -> "Testing":
        """Kanei malakies."""
        return self.__class__

    @property
    @deprecated(message="This does not exist anymore", since="1.2.3")
    def poutous(self) -> dict:
        """Kanei poutous."""
        return self.__dict__


bout = Testing()

print(bout.malakies, bout.poutous)


@deprecated()
def fuck():
    a = 12
    b = 135
    return a + b


print(fuck())


@deprecated(message="rejected section was removed from STELA")
class RejectedItem(t.NamedTuple):
    """Struct containing rejected speech."""

    transcription: Path
    word_frequencies: Path


bah = RejectedItem(transcription=Path.cwd(), word_frequencies=Path.home())
print(bah)
