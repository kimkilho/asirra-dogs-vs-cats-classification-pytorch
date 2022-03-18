# Reference 1: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets

import os.path as osp

from azureml.core import (
    Workspace, Dataset,
)
from azureml.data.datapath import DataPath

SRC_DATA_ROOT_DIR = "C:\\Users\\user\\PycharmProjects\\asirra-dogs-vs-cats-classification-pytorch\\asirra"


if __name__ == '__main__':
    # Configure workspace
    workspace = Workspace.from_config(path='./ws-config.json')
    datastore = workspace.get_default_datastore()

    src_data_dir = osp.join(SRC_DATA_ROOT_DIR, '.')    # FIXME
    asirra_ds = Dataset.File.upload_directory(
        src_dir=src_data_dir,
        target=DataPath(datastore, 'asirra'),
        show_progress=True,
    )

