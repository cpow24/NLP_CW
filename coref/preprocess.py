import numpy as np
import pandas as pd
import os

def PullFilenames(path, filetype):
    folder = os.fsencode(path)
    filenames = []

    # Create list of filenames of interest
    for file in os.listdir(folder):
        filename = os.fsdecode(file)

        if filename.endswith(filetype):
            filenames.append(filename)
    
    return filenames

# def RemoveUndefined(path, filenames):
#     for filename in filenames:
#         print("Processing: ", filename)
#         location = '%s/%s' % (path, filename)
#         with open(location) as file:
#             for line in file:
#                 if '“' in line:
#                     processed_line = line.replace('“', '"')
#                 elif '”' in line:
#                     processed_line = line.replace('”', '"')
#     return

def ProcessFiles(path, filetype):
    filenames = PullFilenames(path, filetype)

    # Run through files of interest
    for filename in filenames:
        processed_file = []
        location = '%s/%s' % (path, filename)
        processed_file_location = location[2:]
        print(processed_file_location)
        with open(location) as file:
            for line in file:
                if filetype == '.conll':
                    if '(' not in line or ')' not in line:
                        continue
                    else:
                        processed_file.append(line)
                else:
                    if 'PER' not in line:
                        continue
                    else:
                        processed_file.append(line)

        with open(processed_file_location, 'x') as f:
            for line in processed_file:
                f.write(line)

    return

# -------------------------------------------------------- #
    
path = 'unprocessed_files/conll'
ProcessFiles(path=path, filetype='.conll')