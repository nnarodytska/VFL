
import argparse
import json

DEBUG100 =  False
def setup_args_load():
    # configuration
    parser = argparse.ArgumentParser(description="Personalization")
    parser.add_argument("--config_info", type=str, help='path to config summary file')
    parser.add_argument("--personalization_steps_replay", type=int, default = None, help='number of steps to be used during personalization')
    parser.add_argument("--concept_epochs", type=int, default = 50, help='number of epochs to be used for learning concept representation')
    parser.add_argument("--concept_lr", type=float, default = None, help='learning rate to be used for learning concept representation')
    parser.add_argument("--personalization_sim_weight", type=float, default = 1.0, help='hyperparameter controlling the importance of the regularization term that maintains the concepts')
    parser.add_argument("--concept_representation", type=str, default = None, help='type of concept representation: decision_tree or linear') # or "decision_tree" or "linear"
    parser.add_argument("--active_layers",nargs="*", type=str, default =None, help='')
    parser.add_argument("--active_concepts",nargs="*", type=str, default = None, help='')
    parser.add_argument("--personalization_lr", type=float, default = None, help='learning rate to be used during personalization')


    args = parser.parse_args()

    with open(args.config_info) as f:
        configs = json.load(f)
    print(configs)
    
    args.active_concepts = args.active_concepts[0].split(',')
    print("Active concepts: ", args.active_concepts)

    if not (args.active_layers  is None):
        args.active_layers = args.active_layers[0].split(',')
        args.active_layers = [int(x) for x in  args.active_layers]
        print("Active layers: ", args.active_layers)

    print(f"reading from data config {configs['data_path']}")
    with open(configs['data_path']) as f:
        # Load the JSON data
        data = json.load(f)
        for k,v in  data.items():
            if k == "personalization_lr" and  not (args.personalization_lr is None):
                continue
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
    
    if args.concept_lr is None:
        args.concept_lr = args.lr
    print(args)
    # exit()
    return args

def setup_args():
    # configuration
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--total_client", type=int, default=100, help='number of clients')
    parser.add_argument("--com_round", type=int, help='number of communication rounds between the server and the clients during federated learning')
    parser.add_argument("--model", type=str, default="mlp", help='model architecture to be used')
    parser.add_argument("--dataset", type=str, default="mnist", help='dataset to be used for experiments: mnist or cub')


    parser.add_argument("--sample_ratio", type=float, help='')
    parser.add_argument("--batch_size", type=int, help='batch size for training and evaluation')
    parser.add_argument("--epochs", type=int, help='number of epochs at the client in each round')
    parser.add_argument("--lr", type=float, help='learning rate at client')
    parser.add_argument("--alpha", type=float, default =None, help='')
    parser.add_argument("--seed", type=int, default =42, help='random seed')
    parser.add_argument("--preprocess", type=bool, default =True, help='')
    parser.add_argument("--cuda", type=bool, default =True, help='is a GPU available for use during training')
    parser.add_argument("--device", type=str, default = "cuda:0", help='the specific GPU to be used if one is available')
    parser.add_argument("--major_classes_num", type=int, default =1, help='')
    parser.add_argument("--augement_data_percent_per_class", type=float, default =0, help='')

    #TODO: Remove / refactor the next two MNIST specific options
    parser.add_argument("--special_data", type=int, default = None, help='')
    parser.add_argument("--augement_data_with_zeros", type=int, default = 0, help='')

    parser.add_argument("--partition", type=str, default ="noniid-#label", help='')
    parser.add_argument("--root_path", type=str, default ='../../datasets', help='path to root folder for storing partitioned dataset, trained model, and config files')
    parser.add_argument("--personalization_steps", type=int, default = 25, help='number of personalization steps')
    parser.add_argument("--personalization_lr", type=float, default = None, help='learning rate to be used during personalization')

    args = parser.parse_args()

    if args.personalization_lr is None:
        args.personalization_lr = args.lr/5


    args.data_path = f"{args.root_path}/{args.dataset}/partition_{args.partition}_major_classes_num_{args.major_classes_num}_clients_{args.total_client}_dir_alpha_{args.alpha}_augement_data_percent_per_class_{args.augement_data_percent_per_class}_augement_data_with_zeros_{args.augement_data_with_zeros}_seed_{args.seed}_special_data_{args.special_data}"
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
                    "models_path",
                    "cuda",
                    "device"]

    for a in  vars(args):
        if a in model_params:
            json_args_model[a] = getattr(args, a)
        else:
            json_args_data[a] = getattr(args, a)

    args.json_args_data = json_args_data
    args.json_args_model = json_args_model

    return args
