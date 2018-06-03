import os
import sys
import argparse
from io import StringIO

sys.path.append("../Libraries")
import azure_tools as azt

DATA_PATH = "../Data/"
SCRIPT_PATH = "./"
LIBRARY_PATH = "../Libraries/"

# CSV_FILES = ["X_train.npy",
#              "y_train.npy",
#              "X_val.npy",
#              "y_val.npy",
#              "X_test.npy"]

CSV_FILES = ["train_features.npy",
             "train_labels.npy",
             "val_features.npy",
             "val_labels.npy",
             "test_features.npy"]


DATA_FILES = [DATA_PATH+fname for fname in CSV_FILES]

SCRIPT_FILES = [SCRIPT_PATH+fname for fname in ["hw6_exercise_2.py", "hw7_exercise_3.py", "sklearnstuff.py"]]
LIBRARY_FILES = [LIBRARY_PATH+fname for fname in \
            ["ml_tools.py", "comp_tools.py", "azure_tools.py"]]

FILES = SCRIPT_FILES + LIBRARY_FILES
#DATA_FILES
import logging

log = logging.getLogger()
if log.hasHandlers():
    log.handlers.clear()
handler = logging.StreamHandler()

formatter = logging.Formatter(
    '%(asctime)s %(name)-8s %(message)s')

handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.DEBUG)

# Pull files from Azure Storage account
def pull_files_from_blob():
    client = azt.get_blob_client()
    for fpath in FILES:
        fname = fpath.split("/")[-1]
        directory = os.path.dirname(fpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        log.info("Uploading file {} to container {}".format(fname, azt.DATA_CONTAINER_NAME))
        client.get_blob_to_path(
            container_name=azt.DATA_CONTAINER_NAME,
            blob_name=fname,
            file_path=fpath)

# Upload files to Azure Storage Account
def upload_files_to_blob():
    client = azt.get_blob_client()
    for fpath in FILES:
        fname = fpath.split("/")[-1]
        log.info("Uploading file {} to container {}".format(fname, azt.DATA_CONTAINER_NAME))
        blob = client.create_blob_from_path(
            container_name=azt.DATA_CONTAINER_NAME,
            blob_name=fname,
            file_path=fpath)

if __name__ == "__main__":
    """Use --upload flag locally and --pull from Azure VM."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pull', dest='pull', action='store_true', required=False)
    parser.add_argument('--upload', dest='upload', action='store_true', required=False)
    parser.set_defaults(pull = False)
    parser.set_defaults(upload = False)
    args = parser.parse_args()
    pull = args.pull
    upload = args.upload

    if upload:
        upload_files_to_blob()
    if pull:
        pull_files_from_blob()
