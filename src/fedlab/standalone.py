from json import load
import os
import random
from copy import deepcopy

import torchvision
import torchvision.transforms as transforms
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("../../")
torch.manual_seed(0)


from mlp import MLP
from standalone_pipeline import EvalPipeline
from partitioned_mnist import PartitionedMNIST
from standalone_setup import setup_args
from basic_client_modifed import SGDSerialClientTrainerExt

from fedlab.contrib.algorithm.basic_server import SyncServerHandler


args = setup_args()
model =MLP(784, 10).cuda()

# server
handler = SyncServerHandler(model = model, 
                            device = 'cuda:0',
                            global_round = args.com_round, 
                            cuda = args.cuda, 
                            sample_ratio = args.sample_ratio)

# client
trainer = SGDSerialClientTrainerExt(model =model, 
                                 num_clients = args.total_client, 
                                 cuda=args.cuda,
                                 device = 'cuda:0')



dataset = PartitionedMNIST( root= args.root_path, 
                            path= args.data_path, 
                            num_clients=args.total_client,
                            dir_alpha=args.alpha,
                            seed=args.seed,
                            preprocess=args.preprocess,
                            partition=args.partition, 
                            major_classes_num= args.major_classes_num,
                            download=True,                           
                            verbose=True,
                            skip_regen = True,
                            transform=transforms.Compose(
                             [transforms.ToPILImage(), transforms.ToTensor()]))

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

test_data = torchvision.datasets.MNIST(root="../../datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

# global main
standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
standalone_eval.main()
standalone_eval.personalize(nb_rounds=args.personalization_steps, save_path= args.models_path, per_lr = args.lr/10, save = True)
