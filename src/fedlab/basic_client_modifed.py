from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from utils import map_inputs_to_rules, calculate_dist_to_rule, calculate_similarity_loss

import torch.nn as nn
import torch

class SGDSerialClientTrainerExt(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, 
                 personalization = False, personalization_rounds = -1, rules = None, sim_weight = 1) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self.personalization_rounds = personalization_rounds
        self.personalization = personalization
        self.rules = rules
        self.sim_weight = sim_weight

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

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        # print(self._model)
        # print(model_parameters.shape)
        self.set_model(model_parameters)
        self._model.train()

        rounds = 0
        #print("----")
        #print(self.model_parameters)
        #print(self.optimizer)

        # map from data points to rules they satisfy
        if not (self.rules is None):
            input_to_rule_map = map_inputs_to_rules(self.model, self.rules, train_loader)

        batch_size = train_loader.batch_size
        for _ in range(self.epochs):
            idx_strt = 0
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                if (self.personalization):
                    assert(self.personalization_rounds != -1)

                    #print(f"personalization_rounds = {rounds}")
                    if rounds >= self.personalization_rounds:
                        #print(self.model_parameters)
                        #exit()
                        return [self.model_parameters]
                    else:
                        output = self.model(data)
                        loss_base = self.criterion(output, target)

                        if self.rules == None:
                            loss = loss_base
                        else:
                            # calculate "distance" of internal representation from rules and corresponding loss
                            latent_vectors = self.model.input_to_representation(data)

                            dist_rep_to_rule = calculate_dist_to_rule(input_to_rule_map[idx_strt:idx_strt+batch_size], latent_vectors, self.rules)
                            loss_sim = calculate_similarity_loss(dist_rep_to_rule)

                            loss = loss_base + self.sim_weight * loss_sim
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                rounds += 1
                idx_strt += batch_size
                        

        return [self.model_parameters]