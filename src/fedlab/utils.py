import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset, BatchSampler
from typing import List, Tuple, Dict
from pathlib import Path

from decision_tree import is_rule_sat, dist_to_rule

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
    targets = dataset.targets
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
    y = np.concatenate((np.ones(subset_size), np.zeros(subset_size)), 0)
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
    for data, target in data_loader:
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
    for idx,latent_vector in latent_vectors:
        rule = rules[input_to_rule_map[idx]]
        dists.append(dist_to_rule(rule, latent_vector))
    return torch.tensor(dists)


def calculate_similarity_loss(dist_rep_to_rule):
    return torch.sum(dist_rep_to_rule)