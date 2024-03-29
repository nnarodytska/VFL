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
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
# from .basic_dataset import FedDataset, Subset
# from ...utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner, MNISTPartitioner

from fedlab.contrib.dataset.basic_dataset import FedDataset, BaseDataset, Subset
from fedlab.utils.dataset.partition import MNISTPartitioner
from fedlab.utils.functional import partition_report
from utils import *


class MNISTPartitionerExt(MNISTPartitioner):
    """Data partitioner for MNIST.

    For details, please check :class:`VisionPartitioner`  and `Federated Dataset and DataPartitioner <FMNISTPartitioner>`_.
    """
    num_features = 784

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
                 target_transform=None,
                 augment_zeros = 0,
                 augment_percent = 0,
                 special_case = None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform
        self.number_classes = 10
        self.augment_percent = augment_percent
        self.augment_zeros = augment_zeros
        self.special_case = special_case


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


    def special_case_zeros_ones(self, dataset, partitioner, is_train):
        print("special_case_zeros_ones")

        if (self.special_case != SPECIAL_CASE_DATA_ZEROS_ONES):
            return
        
        for cid in range(self.num_clients):
            extra_client_dict = {}
            extra_client_dict[cid] = []
        indices =  np.arange(len(dataset.targets))
        # sort sample indices according to labels
        indices_targets = np.vstack((indices, dataset.targets))
        indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
        # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
        #sorted_indices = indices_targets[0, :]
        #print(indices_targets)
        zeros_client = math.floor(self.num_clients/2)
        ones_clients = self.num_clients - zeros_client
        classid_zero = 0
        classid_one = 1
        for cid in range(zeros_client):
            nb_instances = len(partitioner.client_dict[cid])
             
            # zeros
            partitioner.client_dict[cid] = (np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid_zero]))[:nb_instances]
            if self.augment_percent > 0:
                augment_nb_instances = max(ceil(self.augment_percent*len(partitioner.client_dict[cid])), 5)

                # augmented with ones
                extra_client_dict[cid] = (np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid_one]))[:augment_nb_instances]
                partitioner.client_dict[cid] = np.unique(np.hstack((partitioner.client_dict[cid] , extra_client_dict[cid])))



        for cid in range(zeros_client,self.num_clients):
            nb_instances = len(partitioner.client_dict[cid])
             
            # ones
            partitioner.client_dict[cid] = (np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid_one]))[:nb_instances]

            if self.augment_percent > 0:
                # augmented with zeros
                augment_nb_instances = max(ceil(self.augment_percent*len(partitioner.client_dict[cid])), 5)

                extra_client_dict[cid] = (np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid_zero]))[:augment_nb_instances]
                partitioner.client_dict[cid] = np.unique(np.hstack((partitioner.client_dict[cid] , extra_client_dict[cid])))


        
    

            #print(is_train, augment_zeros)

        return
    
    def augment_with_zeros(self, dataset, partitioner, is_train):
        print(f"augment_zeros {self.augment_zeros}")
        if (self.augment_zeros <=0):
            return
        for cid in range(self.num_clients):
            extra_client_dict = {}
            extra_client_dict[cid] = []
        indices =  np.arange(len(dataset.targets))
        # sort sample indices according to labels
        indices_targets = np.vstack((indices, dataset.targets))
        indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
        # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
        #sorted_indices = indices_targets[0, :]
        #print(indices_targets)
    
        for cid in range(self.num_clients):

            #print(augment_nb_instances)
            augment_zeros = copy.deepcopy(self.augment_zeros)
            if not is_train:
                augment_zeros = int(augment_zeros/10)              

            for classid in range(self.number_classes):

                if classid == 0 and augment_zeros > 0 : 
                    extra_client_dict[cid] = (np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid]))[:augment_zeros]
                    #print(indices_targets[0,indices_targets[1,:] ==classid])
                    #print(np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid]))
                    partitioner.client_dict[cid] = np.unique(np.hstack((partitioner.client_dict[cid] , extra_client_dict[cid])))
            
    
    def augment_all_classes(self, dataset, partitioner, is_train):
        if (self.augment_percent <= 0):
            return
        for cid in range(self.num_clients):
            extra_client_dict = {}
            extra_client_dict[cid] = []
        indices =  np.arange(len(dataset.targets))
        # sort sample indices according to labels
        indices_targets = np.vstack((indices, dataset.targets))
        indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
        # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
        #sorted_indices = indices_targets[0, :]
        #print(indices_targets)
    
        for cid in range(self.num_clients):
            augment_nb_instances = max(ceil(self.augment_percent*len(partitioner.client_dict[cid])), 5)
            #print(augment_nb_instances)
        

            for classid in range(self.number_classes):
                extra_client_dict[cid] = (np.random.permutation(indices_targets[0,indices_targets[1,:] ==classid]))[:augment_nb_instances]
                #print(len(extra_client_dict[cid]), cid, classid)
                partitioner.client_dict[cid] = np.unique(np.hstack((partitioner.client_dict[cid] , extra_client_dict[cid])))
            # addign zeros
        
    

            #print(is_train, augment_zeros)


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

            csv_file = f"{self.path}/reports/{pref}{type_data}_mnist_{partition}-label_{major_classes_num}_clients_{self.num_clients}.csv"

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
            plt.savefig(f"{self.path}/reports/{pref}{type_data}_mnist_{partition}-label_{major_classes_num}_clients_{self.num_clients}.png", 
                        dpi=400, bbox_inches = 'tight')

        


        def prepare_data(is_train = True):
            type_data = "train"
            if not is_train:
                type_data = "test"

            dataset = torchvision.datasets.MNIST(root=self.root,
                                                    train=is_train,
                                                    download=download)
            partitioner = MNISTPartitionerExt(dataset.targets,
                                            num_clients=self.num_clients,
                                            partition=partition,
                                            dir_alpha=dir_alpha,
                                            verbose=verbose,
                                            major_classes_num = major_classes_num,
                                            seed=seed)


            if (self.augment_percent > 0):
                self.augment_all_classes(dataset, partitioner, is_train)
            if (self.augment_zeros > 0):
                self.augment_with_zeros(dataset, partitioner, is_train)
                #SPECIAL_CASE_DATA_ZEROS_ONES
            if self.special_case == SPECIAL_CASE_DATA_ZEROS_ONES:
                self.special_case_zeros_ones(dataset, partitioner, is_train)


            # print(indices_targets)
            # exit()


            stats(dataset, type_data, partitioner)
            # partition

            # all data 
            data_indices = []
            for cid in range(self.num_clients):
                data_indices = np.unique(np.hstack((data_indices, partitioner.client_dict[cid]))).astype(int)
            data_indices = list(data_indices)

            all_data = Subset(dataset,
                            data_indices,
                            transform=transform,
                            target_transform=target_transform)
            pref = "all_"
            partitioner.alldata = {0:data_indices}
            
            stats(dataset, type_data, partitioner, pref = pref)

            torch.save(
                    all_data,
                        os.path.join(self.path, type_data, "{}_data.pkl".format(type_data)))

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
