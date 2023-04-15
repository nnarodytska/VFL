import random
import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, BatchSampler
from typing import List, Tuple, Dict
from pathlib import Path
from fedlab.utils.functional import AverageMeter

from mlp import MLP, SmallMLP, TinyMLP
from decision_tree import is_rule_sat, dist_to_rule

def get_model(args):
    if args.model == "mlp":
        return MLP(784, 10).cuda()

    if args.model == "smallmlp":
        return SmallMLP(784, 10).cuda()

    if args.model == "tinymlp":
        return TinyMLP(784, 10).cuda()
    
def subsample_trainset (dataset, fraction = 0.1):

    subsets = []
    for cid in range(dataset.num_clients):
        data4cid = dataset.get_dataloader(cid)
        data4cid.shuffle = True
        nb_samples = len(data4cid.dataset)
        nb_sub_samples = int(nb_samples*fraction)
        subset = random.sample(range(nb_samples), nb_sub_samples)
        subsets.append(torch.utils.data.Subset(data4cid.dataset,subset))
    subsample_train = torch.utils.data.ConcatDataset(subsets)
    print(f"Generated subsampled dataset with fraction {fraction} is of length: {len(subsample_train)}")
    return subsample_train

def generate_concept_dataset(dataset: Dataset, concept_classes: List[int], subset_size: int,
                                   random_seed: int) -> Tuple:
    """
    Return a concept dataset with positive/negatives for MNIST
    Args:
        dataset: the underlying dataset
        random_seed: random seed for reproducibility
        subset_size: size of the positive and negative subset
        concept_classes: the classes where the concept is present

    Returns:
        a concept dataset of the form X (features),y (concept labels)
    """
    dataloader = torch.utils.data.DataLoader(dataset)
    targets = []
    for _,target in dataloader:
        targets.append(target[0].int())
    mask = torch.zeros(len(targets))
    for idx, target in enumerate(targets):  # Scan the dataset for valid examples
        if target in concept_classes:
            mask[idx] = 1
    positive_idx = torch.nonzero(mask).flatten()
    negative_idx = torch.nonzero(1 - mask).flatten()
    positive_loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size,
                                                  sampler=SubsetRandomSampler(positive_idx))
    negative_loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size,
                                                  sampler=SubsetRandomSampler(negative_idx))
    positive_images, positive_labels = next(iter(positive_loader))
    negative_images, negative_labels = next(iter(negative_loader))
    X = np.concatenate((positive_images.cpu().numpy(), negative_images.cpu().numpy()), 0)
    y = np.concatenate((np.ones(len(positive_images)), np.zeros(len(negative_images))), 0)
    np.random.seed(random_seed)
    rand_perm = np.random.permutation(len(X))
    return X[rand_perm], y[rand_perm]


def map_inputs_to_rules(model, rules, data_loader):
    """
    Return a map from an index i in data_loader to the index j in the list of rules such that rules[j] is
      satisfied by data_loader[i]
    Args:
        model: trained DNN
        rules: list of rules where each rule is a triple of the form (class, neuron_ids, neuron_sig)
        data_loader: DataLoader containing the dataset

    Returns:
        an input to rule map
    """
    input_to_rule_map = []
    gpu = next(model.parameters()).device
    for data, target in data_loader:
        data = data.to(gpu)
        target = target.to(gpu)        
        latent_vectors = model.input_to_representation(data)
        for latent_vector in latent_vectors:
            matching_rule = False
            for idx_rule, rule in enumerate(rules):
                if is_rule_sat(rule, latent_vector):
                    input_to_rule_map.append(idx_rule)
                    matching_rule = True
                    break
            if not matching_rule:
                input_to_rule_map.append(-1)
    
    return input_to_rule_map

def calculate_dist_to_rule(input_to_rule_map, latent_vectors, rules):
    dists = []
    for idx,latent_vector in enumerate(latent_vectors):
        if input_to_rule_map[idx] == -1:
            dists.append(torch.tensor(0.0, device=latent_vector.device))
        else:
            rule = rules[input_to_rule_map[idx]]
            dists.append(dist_to_rule(rule, latent_vector))
    return torch.stack(dists)


def calculate_similarity_loss(dist_rep_to_rule):
    return torch.sum(dist_rep_to_rule)

def evaluate_rules(model, rules, data_loader):
    rule_sat_cnt = [0] * len(rules)
    gpu = next(model.parameters()).device
    for data, target in data_loader:
        data = data.to(gpu)
        target = target.to(gpu)
        latent_vectors = model.input_to_representation(data)
        for latent_vector in latent_vectors:
            for idx_rule, rule in enumerate(rules):
                if is_rule_sat(rule, latent_vector):
                    rule_sat_cnt[idx_rule] += 1
                    break
    
    return rule_sat_cnt

def evaluate_label_specific(model, test_loader):
    """Evaluate classify task model accuracy.
    
    Returns:
        list of accuracies for each label
    """
    model.eval()
    gpu = next(model.parameters()).device

    last_layer = list(model.children())[-1]
    correct = [0] * last_layer.out_features
    total = [0] * last_layer.out_features

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i]
                correct[label] += (predicted[i] == label).item()
                total[label] += 1

    accuracy = [correct[i] / total[i] for i in range(len(correct))]
    return accuracy