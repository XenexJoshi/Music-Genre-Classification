import os
import math
import json
import librosa

DATA_PATH = "data" # Directory name of the file containing the .wav files
JSON_PATH = "data.json" # Filename where the encoded data is written

SAMPLE_RATE = 22500 # Standard sampling rate for music classification
DURATION = 30 # GTZAN dataset length
TOTAL_SAMPLE = SAMPLE_RATE * DURATION

def generate_features(data_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segment = 5):
  """
  generate_features(data_path, json_path, n_mfcc, n_fft, hop_length, num_segment) 
  generates a dictionary containing the MFCC and associated labels for the .wav
  files contained in sub-folders within the data_path folder, and writes all the
  collected informations in the json file denoted by json_path. The MFCC is generated
  with arguments n_mfcc, n_fft, hop_length, where each 30s .wav file is split into
  num_segment segments.
  """

  # Dictionary data-type to encode information extracted from .wav file
  data = {
    "mapping": [],
    "mfcc": [],
    "labels": []
  }

  # Duration of a segment of the sample dependent on argument num_segment
  per_segment_sample = int(TOTAL_SAMPLE / num_segment)
  vector_count = math.ceil(per_segment_sample / hop_length)

  # Looping through all genres
  for i, (dir_path, dir_name, file_name) in enumerate(os.walk(data_path)):

    # Saving genre label from directory path
    if dir_path is not data_path:
      path_components = dir_path.split("/")
      genre_label = path_components[-1]
      data["mapping"].append(genre_label)

      print("Processing {}\n".format(genre_label))
    
    # Processing by genre
    for f in file_name:
      # loading an individual file
      file_path = os.path.join(dir_path, f)
      signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

      # Generating segments to extract mfcc
      for j in range(num_segment):
        start = per_segment_sample * j
        end = start + per_segment_sample

        # Generating mfcc features from each segment for each music file
        mfcc = librosa.feature.mfcc(y = signal[start : end],
                                    sr = sample_rate,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)

        # Transforming the mfcc vector in a more processable format
        mfcc = mfcc.T

        # Store mfcc if its dimension matches the expected size
        if len(mfcc) == vector_count:
          data["mfcc"].append(mfcc.tolist())
          data["labels"].append(i - 1)
  
  # Accessing the json file for writing collected data
  with open(json_path, "w") as fp:
    json.dump(data, fp, indent = 4)

# Initiating the data collection process
if __name__ == "__main__":
  generate_features(DATA_PATH, JSON_PATH, num_segment = 10)