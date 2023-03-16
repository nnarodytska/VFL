# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

import pandas as pd
# from .basic_dataset import FedDataset, Subset
# from ...utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner, MNISTPartitioner

from fedlab.contrib.dataset.basic_dataset import FedDataset, BaseDataset, Subset
from fedlab.utils.dataset.partition import MNISTPartitioner
from fedlab.utils.functional import partition_report

class PartitionedMNIST(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        partition (str, optional): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 root,
                 path,
                 num_clients,
                 download=True,
                 preprocess=False,
                 partition="iid",
                 major_classes_num =1,
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 skip_regen = False,
                 transform=None,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform
        self.number_classes = 10


        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            skip_regen = skip_regen,
                            transform=transform,
                            major_classes_num = major_classes_num,
                            target_transform=target_transform)

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   skip_regen =False,
                   transform=None,
                   major_classes_num = 1,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "test"))
            os.mkdir(os.path.join(self.path, "reports"))
            os.mkdir(os.path.join(self.path, "models"))
        else:
            if skip_regen:
                return


        def stats(dataset, type_data, partitioner):
            # debugging 
            csv_file = f"{self.path}/reports/{type_data}_mnist_{partition}-label_{major_classes_num}_clients_{self.num_clients}.csv"
            partition_report(dataset.targets, partitioner.client_dict, 
                            class_num=self.number_classes, 
                            verbose=False, file=csv_file)



            noniid_major_label_part_df = pd.read_csv(csv_file,header=1)
            noniid_major_label_part_df = noniid_major_label_part_df.set_index('client')

            col_names = [f"class{i}" for i in range(self.number_classes)]

            for col in col_names:
                noniid_major_label_part_df[col] = (noniid_major_label_part_df[col] * noniid_major_label_part_df['Amount']).astype(int)

            # select first 10 clients for bar plot
            noniid_major_label_part_df[col_names].plot.barh(stacked=True)  
            # plt.tight_layout()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('sample num')
            plt.savefig(f"{self.path}/reports/{type_data}_mnist_{partition}-label_{major_classes_num}_clients_{self.num_clients}.png", 
                        dpi=400, bbox_inches = 'tight')


        def prepare_data(is_train = True):
            type_data = "train"
            if not is_train:
                type_data = "test"

            dataset = torchvision.datasets.MNIST(root=self.root,
                                                    train=is_train,
                                                    download=download)
            partitioner = MNISTPartitioner(dataset.targets,
                                            num_clients=self.num_clients,
                                            partition=partition,
                                            dir_alpha=dir_alpha,
                                            verbose=verbose,
                                            major_classes_num = major_classes_num,
                                            seed=seed)



            stats(dataset, type_data, partitioner)
            # partition
            subsets = {
                cid: Subset(dataset,
                            partitioner.client_dict[cid],
                            transform=transform,
                            target_transform=target_transform)
                for cid in range(self.num_clients)
            }
            for cid in subsets:
                torch.save(
                    subsets[cid],
                        os.path.join(self.path, type_data, "data{}.pkl".format(cid)))
        prepare_data(is_train = True)
        prepare_data(is_train = False)

    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
