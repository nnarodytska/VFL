import json
import os
from datetime import datetime
import sys

import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

sys.path.append("../../")
torch.manual_seed(0)

from standalone_pipeline import EvalPipeline
from partitioned_mnist import PartitionedMNIST
from partitioned_cub import PartitionedCUB
from setup import setup_args
from basic_client_modifed import SGDSerialClientTrainerExt
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from architectures_mnist import get_mnist_model
from architectures_cub import get_cub_model

def train():
    args = setup_args()
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
                                    device = args.device)

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
                                    augment_percent=args.augement_data_percent_per_class,
                                    augment_zeros = args.augement_data_with_zeros,
                                    special_case = args.special_data,
                                    transform=transforms.Compose(
                                    [transforms.ToPILImage(), transforms.ToTensor()]))
    elif args.dataset == "cub":
        dataset = PartitionedCUB( root= args.root_path, 
                                    path= args.data_path, 
                                    num_clients=args.total_client,
                                    dir_alpha=args.alpha,
                                    seed=args.seed,
                                    preprocess=args.preprocess,
                                    partition=args.partition, 
                                    major_classes_num= args.major_classes_num,
                                    verbose=True,
                                    skip_regen = True,
                                    augment_percent=args.augement_data_percent_per_class)

    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    full_test_data = dataset.get_full_dataset(type = "test")
    full_test_loader = DataLoader(full_test_data, batch_size =  args.batch_size)

    # global main
    standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=full_test_loader)
    standalone_eval.main()

    trainer.setup_optim(args.epochs, args.batch_size, args.lr/10)
    standalone_eval.personalize(nb_rounds=args.personalization_steps, save_path= args.models_path,  save = True)

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)
    jfile = os.path.join(args.data_path, 'config_data.json')
    with open(jfile, 'w') as fp:
        json.dump(args.json_args_data, fp)

    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path, exist_ok=True)
    jfile = os.path.join(args.models_path, 'config_model.json')
    with open(jfile, 'w') as fp:
        json.dump(args.json_args_model, fp)

    json_shortcut = {}
    json_shortcut["models_path"] =  os.path.join(args.models_path, 'config_model.json')
    json_shortcut["data_path"] =  os.path.join(args.data_path, 'config_data.json')
    now = datetime.now() 

    if not os.path.exists(f"{args.root_path}/{args.dataset}/exps_shortcuts/"):
        os.makedirs(f"{args.root_path}/{args.dataset}/exps_shortcuts/", exist_ok=True)    
    shortcut_name = f"{args.root_path}/{args.dataset}/exps_shortcuts/config_{now.strftime('%m-%d-%Y-%H-%M-%S')}.json"
    print(shortcut_name)
    with open(shortcut_name, 'w') as fp:
        json.dump(json_shortcut, fp)


if __name__ == "__main__":
    train()

