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

import copy
import json
from math import ceil
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from fedlab.contrib.dataset.basic_dataset import FedDataset, BaseDataset
from fedlab.utils.dataset.partition import VisionPartitioner
from fedlab.utils.functional import partition_report
from utils import *
from dataset import CUBDataset, load_cub_dataset


class CUBSubset(Dataset):
    """For data subset with different augmentation for different client.

    Args:
        dataset (Dataset): The whole Dataset
        indices (List[int]): Indices of sub-dataset to achieve from ``dataset``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, dataset, indices):
        self.indices = indices
        self.source_dataset = dataset

    def __getitem__(self, index):
        """Get item

        Args:
            index (int): index

        Returns:
            (image, target) where target is index of the target class.
        """
        return self.source_dataset[index]

    def __len__(self):
        return len(self.source_dataset)

class CUBPartitionerExt(VisionPartitioner):
    """Data partitioner for CUB.

    For details, please check :class:`VisionPartitioner`  and `Federated Dataset and DataPartitioner.
    """
    num_features = 299 * 299 * 3
    num_classes = 200

class PartitionedCUB(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
    Args:
        root (str): Path to parent of folder containing datasets.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
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
                 preprocess=False,
                 partition="iid",
                 major_classes_num =1,
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 skip_regen = False,
                 augment_percent = 0) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.number_classes = 200
        self.augment_percent = augment_percent

        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            skip_regen = skip_regen,
                            major_classes_num = major_classes_num)
            
    
    def augment_all_classes(self, dataset, partitioner, is_train):
       raise NotImplementedError

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   skip_regen =False,
                   major_classes_num = 1):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        train_path = f'{self.root}/cub/CUB_processed/class_attr_data_10/train.pkl'
        val_path = f'{self.root}/cub/CUB_processed/class_attr_data_10/val.pkl'
        test_path = f'{self.root}/cub/CUB_processed/class_attr_data_10/test.pkl'
        img_dir = f'{self.root}/cub/CUB_200_2011'
        
        if os.path.exists(self.path) is not True or \
            os.path.exists(os.path.join(self.path, "train")) is not True or \
            len(os.listdir(os.path.join(self.path, "train"))) == 0  or \
            os.path.exists(os.path.join(self.path, "test")) is not True or \
            len(os.listdir(os.path.join(self.path, "test"))) == 0 :
            os.makedirs(self.path,exist_ok=True)
            os.makedirs(os.path.join(self.path, "train"),exist_ok=True)
            os.makedirs(os.path.join(self.path, "test"),exist_ok=True)
            os.makedirs(os.path.join(self.path, "reports"),exist_ok=True)
            os.makedirs(os.path.join(self.path, "models"),exist_ok=True)

        else:
            if skip_regen:
                print(f"we skipped generation: {self.path}")
                return


        def stats(dataset, type_data, partitioner, pref = ""):
            # debugging 
            if len(pref) > 0:
                indices = partitioner.alldata
            else:
                indices = partitioner.client_dict

            csv_file = f"{self.path}/reports/{pref}{type_data}_cub_{partition}-label_{major_classes_num}_clients_{self.num_clients}.csv"

            partition_report(dataset.targets, indices , 
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
            plt.savefig(f"{self.path}/reports/{pref}{type_data}_cub_{partition}-label_{major_classes_num}_clients_{self.num_clients}.png", 
                        dpi=400, bbox_inches = 'tight')

        


        def prepare_data(is_train = True):
            type_data = "train"
            if not is_train:
                type_data = "test"
                raw_data_paths = [test_path]
            else:
                raw_data_paths = [train_path, val_path]
            
            dataset = load_cub_dataset(raw_data_paths, use_attr=False, no_img=False, uncertain_label=False, image_dir=img_dir, n_class_attr=2)
            partitioner = CUBPartitionerExt(dataset.targets,
                                            num_clients=self.num_clients,
                                            partition=partition,
                                            dir_alpha=dir_alpha,
                                            verbose=verbose,
                                            major_classes_num = major_classes_num,
                                            seed=seed)


            if (self.augment_percent > 0):
                self.augment_all_classes(dataset, partitioner, is_train)

            # print(indices_targets)
            # exit()


            stats(dataset, type_data, partitioner)
            # partition

            # all data 
            data_indices = []
            for cid in range(self.num_clients):
                data_indices = np.unique(np.hstack((data_indices, partitioner.client_dict[cid]))).astype(int)
            data_indices = list(data_indices)

            all_data = CUBSubset(dataset,
                            data_indices)
            pref = "all_"
            partitioner.alldata = {0:data_indices}
            
            stats(dataset, type_data, partitioner, pref = pref)

            torch.save(
                    all_data,
                        os.path.join(self.path, type_data, "{}_data.pkl".format(type_data)))

            subsets = {
                cid: CUBSubset(dataset,
                            partitioner.client_dict[cid])
                for cid in range(self.num_clients)
            }


            for cid in subsets:
                torch.save(
                    subsets[cid],
                        os.path.join(self.path, type_data, "data{}.pkl".format(cid)))
        prepare_data(is_train = True)
        prepare_data(is_train = False)

    def get_full_dataset(self, type="train"):
        dataset = torch.load(
                    os.path.join(self.path, type, "{}_data.pkl".format(type)))
        return dataset

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
