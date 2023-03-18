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
from standalone_setup import setup_args
from basic_client_modifed import SGDSerialClientTrainerExt
from dataset import generate_mnist_concept_dataset
from decision_tree import get_invariant, validate


from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.utils.functional import evaluate
from torch import nn

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

data_dir = Path.cwd()/"../../datasets/mnist"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

concept_to_class = {
    "Loop": [0, 2, 6, 8, 9],
    "Vertical Line": [1, 4, 7],
    "Horizontal Line": [4, 5, 7],
    "Curvature": [0, 2, 3, 5, 6, 8, 9],
}
# Load concept sets
X_train, C_train = generate_mnist_concept_dataset(concept_to_class["Curvature"],
                                               data_dir,
                                               train=True,
                                               subset_size=10000, 
                                               random_seed=42)
X_test, C_test = generate_mnist_concept_dataset(concept_to_class["Curvature"],
                                               data_dir,
                                               train=False,
                                               subset_size=1000, 
                                               random_seed=42)

# Evaluate latent representation
X_train = torch.from_numpy(X_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
H_train = handler.model.input_to_representation(X_train)
H_test = handler.model.input_to_representation(X_test)

invariants = get_invariant(H_train.detach().cpu().numpy(), C_train)

print(f'{len(invariants[True])} for concept "Curvature" presented, {len(invariants[False])} for concept "Curvature" not presented')
print(f'{invariants[True][0][-1]} of {int(sum(C_train))} positive samples support concept presented')
print(f'{invariants[False][0][-1]} of {int(len(C_train) - sum(C_train))} negative samples support not presented')

validate(invariants[True][0], True, H_test, C_test)
validate(invariants[False][0], False, H_test, C_test)


#standalone_eval.personalize(nb_rounds=args.personalization_steps, save_path= args.models_path, per_lr = args.personalization_lr, save = False)

