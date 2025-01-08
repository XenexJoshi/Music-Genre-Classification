import os
import math
import json
import librosa

DATA_PATH = "data"
JSON_PATH = "data.json"

SAMPLE_RATE = 22500
DURATION = 30 #GTZAN dataset
TOTAL_SAMPLE = SAMPLE_RATE * DURATION
def generate_features(data_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segment = 5):

  #dictionary data-type to encode information extracted from .wav file
  data = {
    "mapping": [],
    "mfcc": [],
    "delta mfcc": [],
    "labels": []
  }

  per_segment_sample = int(TOTAL_SAMPLE / num_segment)
  vector_count = math.ceil(per_segment_sample / hop_length)

  # looping through genres
  for i, (dir_path, dir_name, file_name) in enumerate(os.walk(data_path)):

    # saving genre label from directory path
    if dir_path is not data_path:
      path_components = dir_path.split("/")
      genre_label = path_components[-1]
      data["mapping"].append(genre_label)

      print("Processing {}\n".format(genre_label))
    
    # process by genre
    for f in file_name:
      # loading an individual file
      file_path = os.path.join(dir_path, f)
      signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

      # generate segments to extract mfcc
      for j in range(num_segment):
        start = per_segment_sample * j
        end = start + per_segment_sample

        mfcc = librosa.feature.mfcc(y = signal[start : end],
                                    sr = sample_rate,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)
        delta_mfcc = librosa.feature.delta(mfcc, order = 1)

        mfcc = mfcc.T
        delta_mfcc = delta_mfcc.T

        # Store mfcc if its dimension matches the expected size
        if len(mfcc) == vector_count:
          data["mfcc"].append(mfcc.tolist())
          data["delta mfcc"].append(delta_mfcc.tolist())
          data["labels"].append(i - 1)
  
  with open(json_path, "w") as fp:
    json.dump(data, fp, indent = 4)

if __name__ == "__main__":
  generate_features(DATA_PATH, JSON_PATH, num_segment = 10)

