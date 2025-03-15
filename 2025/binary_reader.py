import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
from collections import namedtuple
import pickle

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


def round3(t):
    return tuple(np.around(np.array(t), 3).tolist())


def read_record(f):
    # Read header fields.
    SaveSec = struct.unpack('<I', f.read(4))[0]
    Moduls0 = struct.unpack('<B', f.read(1))[0]
    Moduls1 = struct.unpack('<B', f.read(1))[0]
    num_modules = Moduls0 + Moduls1

    # Create a list of dictionaries to store each module's data.
    modules_data = [dict() for _ in range(num_modules)]

    # For each field, read data for all modules.
    # PAC: 3 floats per module, 12 bytes each.
    for i in range(num_modules):
        modules_data[i]['PAC'] = round3(struct.unpack('<3f', f.read(12)))
    
    # QAC: 3 floats per module.
    for i in range(num_modules):
        modules_data[i]['QAC'] = round3(struct.unpack('<3f', f.read(12)))
    
    # PAkku: 2 floats per module.
    for i in range(num_modules):
        modules_data[i]['PAkku'] = round3(struct.unpack('<2f', f.read(8)))
    
    # UAkku: 2 floats per module.
    for i in range(num_modules):
        modules_data[i]['UAkku'] = round3(struct.unpack('<2f', f.read(8)))
    
    # PSol: 2 floats per module.
    for i in range(num_modules):
        modules_data[i]['PSol'] = round3(struct.unpack('<2f', f.read(8)))
    
    # USol: 2 floats per module.
    for i in range(num_modules):
        modules_data[i]['USol'] = round3(struct.unpack('<2f', f.read(8)))
    
    # TAkku: 1 float per module.
    for i in range(num_modules):
        modules_data[i]['TAkku'] = struct.unpack('<f', f.read(4))[0]
    
    # NTC: 2 floats per module.
    for i in range(num_modules):
        modules_data[i]['NTC'] = struct.unpack('<2f', f.read(8))
    
    # ErrAC: 1 unsigned char per module.
    for i in range(num_modules):
        modules_data[i]['ErrAC'] = struct.unpack('<B', f.read(1))[0]
    
    # StatAC: 1 unsigned char per module.
    for i in range(num_modules):
        modules_data[i]['StatAC'] = struct.unpack('<B', f.read(1))[0]
    
    # ErrDC: 1 unsigned char per module.
    for i in range(num_modules):
        modules_data[i]['ErrDC'] = struct.unpack('<B', f.read(1))[0]
    
    # StatDC: 1 unsigned char per module.
    for i in range(num_modules):
        modules_data[i]['StatDC'] = struct.unpack('<B', f.read(1))[0]
    
    # Convert each dictionary to a ModuleData namedtuple.
    modules = [ModuleData(**mod) for mod in modules_data]

    # Read the remaining fields of the record.
    uRMS = round3(struct.unpack('<7f', f.read(28)))
    iRMS = struct.unpack('<6f', f.read(24))
    Pac = struct.unpack('<2f', f.read(8))
    Eac = struct.unpack('<2f', f.read(8))
    EACSum = struct.unpack('<3f', f.read(12))
    EAkkuSum = struct.unpack('<3f', f.read(12))
    ESolSum = struct.unpack('<3f', f.read(12))
    
    # Return the complete BinaryRecord.
    return BinaryRecord(
        SaveSec=SaveSec,
        Moduls0=Moduls0,
        Moduls1=Moduls1,
        modules=modules,
        uRMS=uRMS,
        iRMS=iRMS,
        Pac=Pac,
        Eac=Eac,
        EACSum=EACSum,
        EAkkuSum=EAkkuSum,
        ESolSum=ESolSum
    )



def read_ad_hoc_file(filename):
    records = []
    with open(filename, 'rb') as f:
        # Header: 5 bytes (e.g., "V123/0")
        header_bytes = f.read(5)
        header = header_bytes.decode('ascii')
        while True:
            try:
                record = read_record(f)
            except struct.error as e:
                break
            records.append(record)
    return records


# Example usage:
if __name__ == "__main__":
    bin_filename, pkl_filename = sys.argv[1:]

    # by convention date is only stored in the filename, its directory/date.bin
    date_string = bin_filename.split("/")[-1].split(".")[0]
    print(date_string)
    exit()
    records = read_ad_hoc_file(bin_filename)
    print("#records", len(records))
    with open(pkl_filename, "wb") as f:
        data = (date_string, records)
        pickle.dump(data, pkl_filename)
    exit()

    # Pretty-print the record to see all fields in order
    from pprint import pprint
    T = 4320
    for record in records[T: T + 5]:
        print("==============")
        pprint(record)
        print()
