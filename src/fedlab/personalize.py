import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

sys.path.append("../../")
torch.manual_seed(0)

from standalone_pipeline import EvalPipeline
from partitioned_mnist import PartitionedMNIST
from setup import setup_args_load
from basic_client_modifed import SGDSerialClientTrainerExt
from decision_tree import get_invariant, validate
from utils import extract_testset, subsample_trainset, \
    learn_linear_concept, evaluate_linear_concept, evaluate, get_device
from dataset import generate_mnist_concept_dataset, mnist_concept_to_class, generate_cub_concept_dataset
from architectures_mnist import get_mnist_model
from architectures_cub import get_cub_model

from fedlab.contrib.algorithm.basic_server import SyncServerHandler


def personalize():
    ## extracting rules
    args = setup_args_load()
    device = get_device(args.cuda, args.device)

    if args.dataset == "mnist":
        model = get_mnist_model(args)
    elif args.dataset == "cub":
        model = get_cub_model(args)

    # server
    handler = SyncServerHandler(model = model, 
                                device = args.device,
                                global_round = args.com_round, 
                                cuda = args.cuda,
                                sample_ratio = args.sample_ratio)

    # client
    trainer = SGDSerialClientTrainerExt(model =model, 
                                    num_clients = args.total_client, 
                                    cuda=args.cuda,
                                    device = args.device,
                                    concept_representation=args.concept_representation)

    if args.dataset == "mnist":
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
        # test_data = torchvision.datasets.MNIST(root="../../datasets/mnist/",
        #                                        train=False,
        #                                        transform=transforms.ToTensor())
        # test_loader = DataLoader(test_data, batch_size=1024)
    elif args.dataset == "cub":
        dataset = None
        raise NotImplementedError


    ################ sample train set #########################
    subsample_dataset = subsample_trainset(dataset, fraction = 0.1)
    #########################

    trainer.setup_dataset(dataset)
   
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

    # Load concept sets
    rules = []
    for idx, concept in enumerate(args.active_concepts):
        if args.dataset == "mnist":
            X_train, C_train = generate_mnist_concept_dataset(subsample_dataset, mnist_concept_to_class[concept],
                                                        subset_size=10000,
                                                        random_seed=42)
            X_test_sub, C_test_sub = generate_mnist_concept_dataset(test_data, mnist_concept_to_class[concept],
                                                        subset_size=1000,
                                                        random_seed=42)
            X_test, C_test = generate_mnist_concept_dataset(test_data, mnist_concept_to_class[concept],
                                                        subset_size=10000,
                                                        random_seed=42)
        elif args.dataset == "cub":
            raise NotImplementedError
            X_train, C_train = generate_cub_concept_dataset(subsample_dataset, int(concept),
                                                        subset_size=10000,
                                                        random_seed=42)
            X_test_sub, C_test_sub = generate_cub_concept_dataset(test_data, int(concept),
                                                        subset_size=1000,
                                                        random_seed=42)
            X_test, C_test = generate_cub_concept_dataset(test_data, int(concept),
                                                        subset_size=10000,
                                                        random_seed=42)

        # Evaluate latent representation
        X_train = torch.from_numpy(X_train).to(device)
        X_test_sub = torch.from_numpy(X_test_sub).to(device)
        X_test = torch.from_numpy(X_test).to(device)
        C_train = torch.from_numpy(C_train).to(device)
        C_test_sub = torch.from_numpy(C_test_sub).to(device)
        C_test = torch.from_numpy(C_test).to(device)

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

        elif args.concept_representation in ["linear", "eval_linear"]:
            learn_linear_concept(args, handler.model, X_train, C_train, idx)
            loss, acc, acc_0, acc_1 = evaluate_linear_concept(args, handler.model, X_train, C_train, idx)
            # {"concept classifier train loss":<45} {loss:.2f}, 
            print(f'{concept:<20} train accuracy {acc:.2f} on subsampled train set,              absence train accuracy {acc_0:.2f}, presence train accuracy {acc_1:.2f}')
            loss, acc, acc_0, acc_1 = evaluate_linear_concept(args, handler.model, X_test_sub, C_test_sub, idx)
            print(f'{concept:<20} test accuracy  {acc:.2f} on subsampled, balanced test set, absence test  accuracy {acc_0:.2f}, presence test  accuracy {acc_1:.2f}')
            loss, acc, acc_0, acc_1 = evaluate_linear_concept(args, handler.model, X_test, C_test, idx)
            print(f'{concept:<20} test accuracy  {acc:.2f} on entire test set,               absence test  accuracy {acc_0:.2f}, presence test  accuracy {acc_1:.2f}')


    #setup_optim needs to be called after learn_linear_concept since the latter changes the requires_grad status of model parameters
    trainer.setup_optim(args.epochs, args.batch_size, args.personalization_lr)
    standalone_eval.personalize(nb_rounds=args.personalization_steps_replay, save_path= args.models_path,  rules=rules, sim_weight=args.personalization_sim_weight, 
                                save = False)


if __name__ == "__main__":
    personalize()
