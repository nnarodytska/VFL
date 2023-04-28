
import argparse
import json


def setup_args_load():
    # configuration
    parser = argparse.ArgumentParser(description="Standalone training example")
    parser.add_argument("--config_info", type=str)
    parser.add_argument("--personalization_steps_replay", type=int, default = None)
    parser.add_argument("--personalization_sim_weight", type=float, default = 1.0)
    args = parser.parse_args()

    with open(args.config_info) as f:
        configs = json.load(f)
    print(configs)
    print(f"reading from data config {configs['data_path']}")
    with open(configs['data_path']) as f:
        # Load the JSON data
        data = json.load(f)
        for k,v in  data.items():
            setattr(args, k, v)

    print(f"reading from model config {configs['models_path']}")
    with open(configs['models_path']) as f:
        # Load the JSON data
        data = json.load(f)
        for k,v in  data.items():
            setattr(args, k, v)
    if args.personalization_lr is None:
        args.personalization_lr = args.lr/5

    if args.personalization_steps_replay is None:
        args.personalization_steps_replay = args.personalization_steps
    
    # print(args)
    # exit()
    return args
def setup_args():
    # configuration
    parser = argparse.ArgumentParser(description="Standalone training example")
    parser.add_argument("--total_client", type=int, default=100)
    parser.add_argument("--com_round", type=int)
    parser.add_argument("--model", type=str, default="mlp")


    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--alpha", type=float, default =None)
    parser.add_argument("--seed", type=int, default =42)
    parser.add_argument("--preprocess", type=bool, default =True)
    parser.add_argument("--cuda", type=bool, default =True)
    parser.add_argument("--major_classes_num", type=int, default =1)
    parser.add_argument("--augement_data_percent_per_class", type=float, default =0)
    parser.add_argument("--augement_data_with_zeros", type=int, default = 0)

    parser.add_argument("--partition", type=str, default ="noniid-#label")
    parser.add_argument("--root_path", type=str, default ='../../datasets/mnist/')
    parser.add_argument("--personalization_steps", type=int, default = 25)
    parser.add_argument("--personalization_lr", type=int, default = None)
    
    parser.add_argument("--special_data", type=int, default = None)



    args = parser.parse_args()

    if args.personalization_lr is None:
        args.personalization_lr = args.lr/5


    args.data_path = f"{args.root_path}/partition_{args.partition}_major_classes_num_{args.major_classes_num}_clients_{args.total_client}_dir_alpha_{args.alpha}_augement_data_percent_per_class_{args.augement_data_percent_per_class}_augement_data_with_zeros_{args.augement_data_with_zeros}_seed_{args.seed}_special_data_{args.special_data}"
    args.models_path = f"{args.data_path}/models/batch_size_{args.batch_size}_com_round_{args.com_round}_epochs_{args.epochs}_lr_{args.lr}_sample_ratio_{args.sample_ratio}_personalization_steps_{args.personalization_steps}_model_{args.model}"
    # print( args.data_path)
    # print( args.models_path)

    json_args_data = {}
    json_args_model = {}

    model_params = ["total_client",
                    "com_round",
                    "model",
                    "sample_ratio",
                    "batch_size",
                    "epochs",
                    "lr",
                    "alpha",
                    "personalization_steps",
                    "models_path"]

    for a in  vars(args):
        if a in model_params:
            json_args_model[a] = getattr(args, a)
        else:
            json_args_data[a] = getattr(args, a)

    args.json_args_data = json_args_data
    args.json_args_model = json_args_model

    return args