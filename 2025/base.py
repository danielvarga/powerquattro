from collections import namedtuple


# Define a namedtuple for a single module's data.
ModuleData = namedtuple("ModuleData", [
    "PAC", "QAC", "PAkku", "UAkku", "PSol", "USol",
    "TAkku", "NTC", "ErrAC", "StatAC", "ErrDC", "StatDC"
])

# Define a top-level namedtuple to hold the complete record.
BinaryRecord = namedtuple("BinaryRecord", [
    "SaveSec", "Moduls0", "Moduls1", "modules",
    "uRMS", "iRMS", "Pac",
    "Eac", "EACSum", "EAkkuSum", "ESolSum"
])
