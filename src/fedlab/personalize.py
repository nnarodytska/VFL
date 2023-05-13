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
from setup import setup_args, setup_args_load
from basic_client_modifed import SGDSerialClientTrainerExt
from decision_tree import get_invariant, validate
from utils import extract_testset, generate_concept_dataset, get_model, subsample_trainset, \
    learn_linear_concept, evaluate_linear_concept, evaluate

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from torch import nn


def personalize():
    ## extracting rules
    args = setup_args_load()
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
                                    device = 'cuda:0',
                                    concept_representation=args.concept_representation)

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
    subsample_dataset = subsample_trainset(dataset, fraction = 0.1)
    #########################

    trainer.setup_dataset(dataset)
    # test_data = torchvision.datasets.MNIST(root="../../datasets/mnist/",
    #                                        train=False,
    #                                        transform=transforms.ToTensor())
    # test_loader = DataLoader(test_data, batch_size=1024)


    test_data = extract_testset(dataset, type = "test")
    test_loader = DataLoader(test_data, batch_size =  args.batch_size)

    standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)

    # load global model and read clients main
    standalone_eval.load_global_model(path = args.models_path)

    ###############################################
    # we have global model here
    ###############################################
    print("Model architecture:")
    print(handler.model)
    # evalution of `global` training set
    loss, acc = evaluate(handler.model, nn.CrossEntropyLoss(), test_loader)
    print("Global model loss {:.4f}, test accuracy {:.4f}".format(loss, acc))

    # concept_to_class = {
    #     "Loop": [0, 2, 6, 8, 9],
    #     "Vertical Line": [1, 4, 7],
    #     "Horizontal Line": [4, 5, 7],
    #     "Curvature": [0, 2, 3, 5, 6, 8, 9],
    # }

    concept_to_class = {
        "Curvature": [0, 2, 3, 5, 6, 8, 9],
        "Loop": [0, 6, 8, 9],
        "Vertical Line": [1, 4, 5, 7],
        "Horizontal Line": [2, 4, 5, 7]
    }

    # Load concept sets
    rules = []
    for idx, concept in enumerate(["Curvature", "Loop", "Vertical Line", "Horizontal Line"]):
        X_train, C_train = generate_concept_dataset(subsample_dataset, concept_to_class[concept],
                                                    subset_size=10000,
                                                    random_seed=42)
        X_test_sub, C_test_sub = generate_concept_dataset(test_data, concept_to_class[concept],
                                                    subset_size=1000,
                                                    random_seed=42)
        X_test, C_test = generate_concept_dataset(test_data, concept_to_class[concept],
                                                    subset_size=10000,
                                                    random_seed=42)

        # Evaluate latent representation
        X_train = torch.from_numpy(X_train).to("cuda:0")
        X_test_sub = torch.from_numpy(X_test_sub).to("cuda:0")
        X_test = torch.from_numpy(X_test).to("cuda:0")
        C_train = torch.from_numpy(C_train).to("cuda:0")
        C_test_sub = torch.from_numpy(C_test_sub).to("cuda:0")
        C_test = torch.from_numpy(C_test).to("cuda:0")

        if args.concept_representation == "decision_tree":
            H_train = handler.model.input_to_representation(X_train)
            H_test = handler.model.input_to_representation(X_test)
            invariants = get_invariant(H_train.detach().cpu().numpy(), C_train)

            print(f'{len(invariants[True])} rules for concept {concept} present, {len(invariants[False])} rules for concept {concept} not present')
            print(f'{invariants[True][0][-1]} of {int(sum(C_train))} positive samples support first rule for concept present')
            print(f'{invariants[False][0][-1]} of {int(len(C_train) - sum(C_train))} negative samples support first rule for concept not present')

            validate(invariants[True][0], True, H_test, C_test)
            validate(invariants[False][0], False, H_test, C_test)

            # We choose to enforce only two rules during personalization; the top rule for each of concept present and absent
            rules.append((True, invariants[True][0][0], invariants[True][0][1]))
            rules.append((False, invariants[False][0][0], invariants[False][0][1]))

        elif args.concept_representation == "linear":
            learn_linear_concept(args, handler.model, X_train, C_train, idx)
            loss, acc, acc_0, acc_1 = evaluate_linear_concept(args, handler.model, X_train, C_train, idx)
            print(f'{concept} concept classifier train loss {loss}, overall train accuracy {acc}, absence train accuracy {acc_0}, presence train accuracy {acc_1}')
            loss, acc, acc_0, acc_1 = evaluate_linear_concept(args, handler.model, X_test_sub, C_test_sub, idx)
            print(f'{concept} concept classifier test loss {loss}, test accuracy {acc} on subsampled, balanced test set, absence train accuracy {acc_0}, presence train accuracy {acc_1}')
            loss, acc, acc_0, acc_1 = evaluate_linear_concept(args, handler.model, X_test, C_test, idx)
            print(f'{concept} concept classifier loss {loss}, test accuracy {acc} on entire test set, absence train accuracy {acc_0}, presence train accuracy {acc_1}')


    #setup_optim needs to be called after learn_linear_concept since the latter changes the requires_grad status of model parameters
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    standalone_eval.personalize(nb_rounds=args.personalization_steps_replay, save_path= args.models_path, 
                                per_lr = args.personalization_lr, rules=rules, sim_weight=args.personalization_sim_weight, 
                                save = False)


if __name__ == "__main__":
    personalize()
