# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/)
# and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    get dataloader for dataset in LEAF processed
"""
import logging
from pathlib import Path
from typing import Any

import torch
from leaf.pickle_dataset import PickleDataset

BASE_DIR = Path(__file__).resolve().parents[2]


def get_LEAF_dataloader(
    dataset: str,
    client_id: int = 0,
    batch_size: int = 128,
    data_root: str = None,
    pickle_root: str = None,
) -> tuple[Any, Any]:
    """Get dataloader with ``batch_size`` param for client with ``client_id``

    Args:
        dataset (str):  dataset name string to get dataloader
        client_id (int, optional): assigned client_id to get dataloader for this client. Defaults to 0
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128
        data_root (str): path for data saving root.
                        Default to None and will be modified to the datasets folder in FedLab: "fedlab-benchmarks/datasets"
        pickle_root (str): path for pickle dataset file saving root.
                        Default to None and will be modified to Path(__file__).parent / "pickle_datasets"
    Returns:
        A tuple with train dataloader and test dataloader for the client with `client_id`

    Examples:
        trainloader, testloader = get_LEAF_dataloader(dataset='shakespeare', client_id=1)
    """
    # Need to run leaf/gen_pickle_dataset.sh to generate pickle files for this dataset firstly
    pdataset = PickleDataset(
        dataset_name=dataset, data_root=data_root, pickle_root=pickle_root
    )
    try:
        trainset = pdataset.get_dataset_pickle(
            dataset_type="train", client_id=client_id
        )
        testset = pdataset.get_dataset_pickle(
            dataset_type="test", client_id=client_id
        )
    except FileNotFoundError:
        logging.error(
            f"""
                        No built PickleDataset json file for {dataset}-client {client_id} in {pdataset.pickle_root.resolve()}
                        Please run `{BASE_DIR / 'leaf/gen_pickle_dataset.sh'} to generate {dataset} pickle files firstly!`
                        """
        )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, drop_last=False
    )  # avoid train dataloader size 0
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), drop_last=False, shuffle=False
    )

    return trainloader, testloader


def get_LEAF_all_test_dataloader(
    dataset: str,
    batch_size: int = 128,
    data_root: str = None,
    pickle_root: str = None,
) -> Any:
    """Get dataloader for all clients' test pickle file

    Args:
        dataset (str): dataset name
        batch_size (int, optional): the number of batch size for dataloader. Defaults to 128
        data_root (str): path for data saving root.
                        Default to None and will be modified to the datasets folder in FedLab: "fedlab-benchmarks/datasets"
        pickle_root (str): path for pickle dataset file saving root.
                        Default to None and will be modified to Path(__file__).parent / "pickle_datasets"
    Returns:
        ConcatDataset for all clients' test dataset
    """
    pdataset = PickleDataset(
        dataset_name=dataset, data_root=data_root, pickle_root=pickle_root
    )

    try:
        all_testset = pdataset.get_dataset_pickle(dataset_type="test")
    except FileNotFoundError:
        logging.error(
            f"""
                        No built test PickleDataset json file for {dataset} in {pdataset.pickle_root.resolve()}
                        Please run `{BASE_DIR / 'leaf/gen_pickle_dataset.sh'} to generate {dataset} pickle files firstly!`
                        """
        )
    test_loader = torch.utils.data.DataLoader(
        all_testset, batch_size=batch_size, drop_last=True
    )  # avoid train dataloader size 0
    return test_loader
