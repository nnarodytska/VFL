from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer

import torch.nn as nn
import torch

class SGDSerialClientTrainerExt(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, 
                 personalization = False, personalization_rounds = -1) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self.personalization_rounds = personalization_rounds
        self.personalization = personalization

    def ok(self):
        print("ok")

    def setup_lr(self,  lr):
        """Set up local optimization configuration.

        Args:
            lr (float): Learning rate.
        """
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        rounds = 0
        #print("----")
        #print(self.model_parameters)
        #print(self.optimizer)
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                rounds += 1
                if (self.personalization):
                    assert(self.personalization_rounds != -1)
                    
                    #print(f"personalization_rounds = {rounds}")
                    if rounds >= self.personalization_rounds:
                        #print(self.model_parameters)
                        #exit()
                        return [self.model_parameters]
                        

        return [self.model_parameters]