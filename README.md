# the-sound-of-music


python3 preprocess.py /path/to/data/ [-f]

python3 train.py "model name"

python3 test.py /path/to/data/

python3 infer.py /path/to/model/ /path/to/outdio/file/ /path/to/midi/output/


see constants.py for TRAIN/TEST_PROCESSED_DIR where preprocessed filed are saved to later be used for training/testing.

The data_dir must contain a csv_file similar to maestro_v2.0.0.csv which constains information about train/test split.


