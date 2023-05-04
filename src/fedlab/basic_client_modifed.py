from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from utils import map_inputs_to_rules, calculate_dist_to_rule, calculate_similarity_loss, evaluate_rules

import torch.nn as nn
import torch

import gurobipy as gp
from gurobipy import GRB

class SGDSerialClientTrainerExt(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, 
                 personalization = False, personalization_rounds = -1, rules = None, sim_weight = 1) -> None:
        super().__init__(model, num_clients, False, None, personal)
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

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)

            self.set_model(model_parameters)
            # map from data points to rules they satisfy
            if not (self.rules is None):
                input_to_rule_map = map_inputs_to_rules(self.model, self.rules, data_loader)
            pack = self.train(model_parameters, data_loader)

            if self.rules != None:
                rule_sat_cnt = evaluate_rules(self.model, self.rules, data_loader)
                print(f'before repair: \
                      client {id}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]}')

            pack = self.repair(model_parameters, data_loader, input_to_rule_map)
            self.cache.append(pack)

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
                        loss = self.criterion(output, target)

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
    

    def repair(self, model_parameters, train_loader, input_to_rule_map):
        """Repair weights for one client using Gurobi.

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

        if self.rules == None:
            return [self.model_parameters]
        
        for data, target in train_loader:
            if self.cuda:
                data = data.cuda(self.device)
                target = target.cuda(self.device)

            if (self.personalization):
                global_w = self.model.fc2.weight.detach().cpu().numpy()
                global_b = self.model.fc2.bias.detach().cpu().numpy()

                gurobi_model = gp.Model("personalization")
                nrow, ncol = global_w.shape
                global_w = global_w.flatten()
                indices = [*range(nrow * ncol)]
                gurobi_w = gurobi_model.addVars(indices, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
                gurobi_model.setObjective(gp.quicksum((gurobi_w[i] - global_w[i]) ** 2 for i in indices),
                                            GRB.MINIMIZE)

                layer2_input = self.model.layer2_input(data).detach().cpu().numpy()
                for i in range(len(data)):
                    rule = self.rules[input_to_rule_map[i]]
                    for id in range(len(rule[1])):
                        constraint = gp.quicksum(
                            layer2_input[i][j] * gurobi_w[j * ncol + id] + global_b[id] for j in range(ncol))
                        op = rule[2][2 * id]
                        vsig = rule[2][2 * id + 1]

                        if op == "<=":
                            gurobi_model.addConstr(constraint <= vsig)
                        else:
                            # gurobi hard to optimize >
                            gurobi_model.addConstr(constraint >= vsig)

                gurobi_model.feasRelaxS(1, True, False, True)
                gurobi_model.optimize()

                optimized_w = torch.tensor([gurobi_w[i].x for i in indices]).reshape(nrow, ncol)

                if self.cuda:
                    optimized_w = optimized_w.cuda(self.device)

                self.model.fc2.weight = nn.Parameter(optimized_w)

        return [self.model_parameters]