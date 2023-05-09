from json import load
import json
import os
import random
from copy import deepcopy
from utils import extract_testset, get_model, subsample_trainset

import torchvision
import torchvision.transforms as transforms
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("../../")
torch.manual_seed(0)


from standalone_pipeline import EvalPipeline
from partitioned_mnist import PartitionedMNIST
from setup import setup_args
from basic_client_modifed import SGDSerialClientTrainerExt

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from datetime import datetime

args = setup_args()


model = get_model(args)
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
                            augment_percent=args.augement_data_percent_per_class,
                            augment_zeros = args.augement_data_with_zeros,
                            special_case = args.special_data,
                            transform=transforms.Compose(
                             [transforms.ToPILImage(), transforms.ToTensor()]))

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# test_data = torchvision.datasets.MNIST(root="../../datasets/mnist/",
#                                        train=False,
#                                        transform=transforms.ToTensor())
# test_loader = DataLoader(test_data, batch_size=1024)

test_data = extract_testset(dataset, type = "test")
test_loader = DataLoader(test_data, batch_size =  args.batch_size)

# global main
standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
standalone_eval.main()

standalone_eval.personalize(nb_rounds=args.personalization_steps, save_path= args.models_path, 
                            per_lr = args.personalization_lr, save = True)

jfile = os.path.join(args.data_path, 'config_data.json')
with open(jfile, 'w') as fp:
    json.dump(args.json_args_data, fp)
jfile = os.path.join(args.models_path, 'config_model.json')
with open(jfile, 'w') as fp:
    json.dump(args.json_args_model, fp)

json_shortcut = {}
json_shortcut["models_path"] = args.models_path
json_shortcut["data_path"] = args.data_path
now = datetime.now() 

shortcut_name = f"../../datasets/mnist/exps_shortcuts/config_{now.strftime('%m-%d-%Y-%H-%M-%S')}"
with open(shortcut_namefile, 'w') as fp:
    json.dump(json_shortcut, fp)




