
import argparse

def setup_args():
    # configuration
    parser = argparse.ArgumentParser(description="Standalone training example")
    parser.add_argument("--total_client", type=int, default=100)
    parser.add_argument("--com_round", type=int)

    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--alpha", type=float, default =None)
    parser.add_argument("--seed", type=int, default =42)
    parser.add_argument("--preprocess", type=bool, default =True)
    parser.add_argument("--cuda", type=bool, default =True)
    parser.add_argument("--major_classes_num", type=int, default =1)
    parser.add_argument("--partition", type=str, default ="noniid-#label")
    parser.add_argument("--root_path", type=str, default ='../../datasets/mnist/')
    parser.add_argument("--personalization_steps", type=int, default = 25)



    args = parser.parse_args()


    args.data_path = f"{args.root_path}/partition_{args.partition}_major_classes_num_{args.major_classes_num}_clients_{args.total_client}_dir_alpha_{args.alpha}_seed_{args.seed}"
    args.models_path = f"{args.data_path}/models/batch_size_{args.batch_size}_com_round_{args.com_round}_epochs_{args.epochs}_sample_ratio_{args.sample_ratio}_personalization_steps_{args.personalization_steps}"

    return args
