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
import matplotlib.pyplot as plt
from setup import DEBUG100

from decision_tree import is_rule_sat, dist_to_rule
from torchvision.utils import save_image
SPECIAL_CASE_DATA_ZEROS_ONES = 100

def get_device(use_cuda, device_name):
    if use_cuda:
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    return device

def extract_testset(dataset, type = "test"):
    return  dataset.get_full_dataset(type = type)

def subsample_trainset(dataset, fraction = 0.1):
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
    model.eval()
    gpu = next(model.parameters()).device
    with torch.no_grad():
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
    model.eval()
    gpu = next(model.parameters()).device
    with torch.no_grad():
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

    # Get the number of output features
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)
            output_shape = model(inputs).shape
            num_outputs = output_shape[-1]
            break
    
    correct = [0] * num_outputs
    total = [0] * num_outputs

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
                

    accuracy = []
    for i in range(len(correct)):
        if ( total[i] > 0):
            accuracy.append(correct[i] / total[i])
        else:
            accuracy.append(0)
    return accuracy

def plot_client_stats(client_stats,id,name):
    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Plot the global and local accuracies as horizontal lines
    ax.axhline(y=client_stats["global_accuracy"], color="blue", linestyle="--", label="Global Accuracy")
    ax.axhline(y=client_stats["local_accuracy"], color="green", linestyle="--", label="Local Accuracy")

    # Plot the label-specific accuracies as bars
    ind = np.arange(10)  # the x-axis locations for the bars
    width = 0.35  # the width of the bars
    ax.bar(ind - width/2, client_stats["label_specific_global_accuracy"], width, color="blue", label="Global Label-Specific Accuracy")
    ax.bar(ind + width/2, client_stats["label_specific_local_accuracy"], width, color="green", label="Local Label-Specific Accuracy")

    # Add axis labels and title
    ax.set_xlabel("Labels")
    ax.set_ylabel("Accuracy")
    ax.set_title(f'Client {id} stats')

    # Add a legend
    ax.legend()

    # Save the plot to disk
    plt.savefig(f'client_stats_{id}_{name}.png')

def learn_linear_concept(args, model, X, Y, concept_id):
    concept_dataset = torch.utils.data.TensorDataset(X,Y)
    concept_dataloader = DataLoader(concept_dataset, batch_size=args.batch_size, shuffle=True)
    for name, param in model.named_parameters():
        #print(f"{name}, {param.requires_grad}, {param}")
        param.requires_grad = False

    for name, module in model.named_modules():
        #print( "~~~~~", name, module,  module == model.concept_layers[concept_id])
        if module == model.concept_layers[concept_id]:
            for pname, params in module.named_parameters():
                params.requires_grad = True
                if DEBUG100 == 1: print(f"grad on {name}-{pname}")
                #print(params,  params.requires_grad )


    # optimizer = torch.optim.SGD(model.concept_layers[concept_id].parameters(), args.lr)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.concept_lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = args.concept_epochs
    model.train()
    model.start_probe_mode()
    for _ in range(epochs):
        for data, target in concept_dataloader:
            outputs = model(data)
            loss = loss_fn(outputs[concept_id+1], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.stop_probe_mode()

def evaluate_linear_concept(args, model, X, Y, concept_id):
    """Evaluate concept representation accuracy.
    
    Returns:
        (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device
    concept_dataset = torch.utils.data.TensorDataset(X,Y)
    concept_dataloader = DataLoader(concept_dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    acc_0_ = AverageMeter()
    acc_1_ = AverageMeter()

    model.start_probe_mode()
    with torch.no_grad():
        for inputs, labels in concept_dataloader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)
            labels_0_idx = torch.nonzero(labels==0)
            labels_1_idx = torch.nonzero(labels==1)

            outputs = model(inputs)
            loss = loss_fn(outputs[concept_id+1], labels)

            _, predicted = torch.max(outputs[concept_id+1], 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))
            acc_0_.update(torch.sum(predicted[labels_0_idx].eq(labels[labels_0_idx])).item(), len(labels_0_idx))
            acc_1_.update(torch.sum(predicted[labels_1_idx].eq(labels[labels_1_idx])).item(), len(labels_1_idx))
    model.stop_probe_mode()
    return loss_.sum, acc_.sum/acc_.count, acc_0_.sum/acc_0_.count, acc_1_.sum/acc_1_.count

def evaluate_linear_concepts(model, data_loader):
    """Evaluate concept representation accuracy.
    
    Returns:
        (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device
    model.start_probe_mode()

    # Get the number of concepts
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(gpu)
            target = target.to(gpu)
            num_concepts = len(model(data)) - 1
            break

    concept_present_cnt = [0] * num_concepts
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(gpu)
            target = target.to(gpu)
            outputs = model(data)
            for concept_id in range(num_concepts):
                _, predicted = torch.max(outputs[concept_id+1], 1)
                concept_present_cnt[concept_id] += torch.sum(predicted).item()
    model.stop_probe_mode()
    return concept_present_cnt

def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy.
    
    Returns:
        (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.sum/acc_.count
