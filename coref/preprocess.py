import numpy as np
import pandas as pd
import os

def PullFilenames(folder, filetype):
    folder = os.fsencode(folder)
    filenames = []

    # Create list of filenames of interest
    for file in os.listdir(folder):
        filename = os.fsdecode(file)

        if filename.endswith('.ann'):
            filenames.append(filename)
    
    return filenames

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
                if 'PER' not in line:
                    continue
                else:
                    processed_file.append(line)

        with open(processed_file_location, 'x') as f:
            for line in processed_file:
                f.write(line)

    return
    
path = 'unprocessed_files/brat'
ProcessFiles(path=path, filetype='.ann')