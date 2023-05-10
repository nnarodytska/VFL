from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from utils import map_inputs_to_rules, calculate_dist_to_rule, calculate_similarity_loss

import torch.nn as nn
import torch

class SGDSerialClientTrainerExt(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, 
                 personalization = False, personalization_rounds = -1, rules = None, sim_weight = 1, 
                 concept_representation = None, global_model = None) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self.personalization_rounds = personalization_rounds
        self.personalization = personalization
        self.rules = rules
        self.sim_weight = sim_weight
        self.concept_representation = concept_representation
        self.global_model = global_model

    def ok(self):
        print("ok")

    def setup_lr(self,  lr):
        """Set up local optimization configuration.

        Args:
            lr (float): Learning rate.
        """
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)

    def setup_rules(self, rules):
        """Set up rules that are to be maintained.

        Args:
            rules (List): List of rules where each rule is a triple of the form (class, neuron ids, neuron signature).
        """
        self.rules = rules

    def setup_sim_weight(self, weight):
        """Set up sim_weight.

        Args:
            weight (float): sim_weight.
        """
        self.sim_weight = weight
    
    def setup_global_model(self, global_model):
        """Set up global_model.

        Args:
            global_model (torch.nn.Module): global_model.
        """
        self.global_model = global_model

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        params = []
        if self.concept_representation == "linear":
            for layer in self._model.pred_layers:
                params += list(layer.parameters())
        else:
            params = self._model.parameters()

        self.optimizer = torch.optim.SGD(params, lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        if self.personalization and self.concept_representation == "decision_tree" and self.rules == None:
            raise Exception("If using decision tree rules for personalization, rules need to be provided.")
        
        if self.personalization and self.concept_representation == "linear" and not hasattr(self._model, "concept_layers"):
            raise Exception("If using linear concepts for personalization, model needs to have concept layers.")
        

        # print(self._model)
        # print(model_parameters.shape)
        self.set_model(model_parameters)

        rounds = 0
        #print("----")
        #print(self.model_parameters)
        #print(self.optimizer)

        # map from data points to rules they satisfy
        # if not (self.rules is None):
        if self.personalization and self.concept_representation == "decision_tree":
            input_to_rule_map = map_inputs_to_rules(self.global_model, self.rules, train_loader)
        
        if self.personalization and self.concept_representation == "linear":
            self.global_model.eval()
            self.global_model.start_probe_mode()
            input_to_concept_labels = []
            with torch.no_grad():
                for data, target in train_loader:
                    if self.cuda:
                        data = data.cuda(self.device)
                        target = target.cuda(self.device)
                    concept_outputs = self.global_model(data)[1:-1]
                    concept_labels = []
                    for c_idx, concept_output in enumerate(concept_outputs):
                        concept_labels.append(torch.unsqueeze(torch.argmax(concept_output, dim=1), dim=-1))
                    input_to_concept_labels.append(torch.cat(concept_labels, dim=1))
                input_to_concept_labels = torch.cat(input_to_concept_labels)
            self.global_model.stop_probe_mode()
                
        self._model.train()
        batch_size = train_loader.batch_size
        # print(self.epochs)
        # exit()
        for _ in range(self.epochs):
            idx_strt = 0
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                if (self.personalization):
                    assert(self.personalization_rounds != -1)
                    # print(f"personalization_rounds = {rounds}")
                    if rounds >= self.personalization_rounds:
                        #print(self.model_parameters)
                        #exit()
                        return [self.model_parameters]
                    else:
                        if self.concept_representation == None:
                            output = self.model(data)
                            loss_base = self.criterion(output, target)
                            loss = loss_base
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        elif self.concept_representation == "decision_tree":
                            output = self.model(data)
                            loss_base = self.criterion(output, target)
                            # calculate "distance" of internal representation from rules and corresponding loss
                            latent_vectors = self.model.input_to_representation(data)
                            dist_rep_to_rule = calculate_dist_to_rule(input_to_rule_map[idx_strt:idx_strt+batch_size], latent_vectors, self.rules)
                            loss_sim = calculate_similarity_loss(dist_rep_to_rule)
                            loss = loss_base + self.sim_weight * loss_sim
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            #print(len(target[target == 0]), len(target[target == 1]), float(loss))

                        elif self.concept_representation == "linear":
                            self._model.start_probe_mode()
                            output = self.model(data)
                            loss_base = self.criterion(output[0], target)
                            concept_labels = output[1:-1]
                            loss_concept = 0
                            for c_idx, concept_label in enumerate(concept_labels):
                                loss_concept += self.criterion(concept_label, input_to_concept_labels[idx_strt:idx_strt+batch_size, c_idx])
                            loss = loss_base + self.sim_weight * loss_concept
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self._model.stop_probe_mode()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                rounds += 1
                idx_strt += batch_size
                        
        return [self.model_parameters]