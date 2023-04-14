from json import load
import os
import random
from copy import deepcopy
from pathlib import Path

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
from setup import setup_args
from basic_client_modifed import SGDSerialClientTrainerExt
from decision_tree import get_invariant, validate
from utils import generate_concept_dataset, get_model, subsample_trainset


from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.utils.functional import evaluate
from torch import nn

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
                            transform=transforms.Compose(
                             [transforms.ToPILImage(), transforms.ToTensor()]))


################ sample train set #########################
subsample_dataloader = subsample_trainset(dataset, fraction = 0.1)
#########################

trainer.setup_dataset(dataset)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

test_data = torchvision.datasets.MNIST(root="../../datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
# laod global  and read clients main

standalone_eval.load_global_model(path = args.models_path)

###############################################
# we have global model here
###############################################
print(handler.model)
# evalution of `global` trainign set
loss, acc = evaluate(handler.model, nn.CrossEntropyLoss(), test_loader)
print("loss {:.4f}, test accuracy {:.4f}".format(loss, acc))


## extracting rules

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_data = torchvision.datasets.MNIST(root="../../datasets/mnist/",
                                       train=True,
                                       transform=transforms.ToTensor())
concept_to_class = {
    "Loop": [0, 2, 6, 8, 9],
    "Vertical Line": [1, 4, 7],
    "Horizontal Line": [4, 5, 7],
    "Curvature": [0, 2, 3, 5, 6, 8, 9],
}
# Load concept sets
X_train, C_train = generate_concept_dataset(train_data, concept_to_class["Curvature"],
                                               subset_size=10000, 
                                               random_seed=42)
X_test, C_test = generate_concept_dataset(train_data, concept_to_class["Curvature"],
                                               subset_size=1000, 
                                               random_seed=42)

# Evaluate latent representation
X_train = torch.from_numpy(X_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
H_train = handler.model.input_to_representation(X_train)
H_test = handler.model.input_to_representation(X_test)

invariants = get_invariant(H_train.detach().cpu().numpy(), C_train)

print(f'{len(invariants[True])} rules for concept "Curvature" present, {len(invariants[False])} rules for concept "Curvature" not present')
print(f'{invariants[True][0][-1]} of {int(sum(C_train))} positive samples support first rule for concept present')
print(f'{invariants[False][0][-1]} of {int(len(C_train) - sum(C_train))} negative samples support first rule for concept not present')

validate(invariants[True][0], True, H_test, C_test)
validate(invariants[False][0], False, H_test, C_test)

# We choose to enforce only two rules during personalization; the top rule for each of concept present and absent 
rules = []
rules.append((True, invariants[True][0][0], invariants[True][0][1]))
rules.append((False, invariants[False][0][0], invariants[False][0][1]))

standalone_eval.personalize(nb_rounds=args.personalization_steps_replay, save_path= args.models_path, 
                            per_lr = args.personalization_lr, rules=rules, sim_weight=args.personalization_sim_weight, save = False)

