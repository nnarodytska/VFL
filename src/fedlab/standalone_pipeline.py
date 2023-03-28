import os
from fedlab.utils.functional import evaluate
from fedlab.core.standalone import StandalonePipeline
import torch
from torch import nn
from utils import evaluate_rules

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
    
    def main(self):
        nb_round = 0
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            nb_round += 1
            print("nb rounds {}: loss {:.4f}, test accuracy {:.4f}".format(nb_round, loss, acc))
            for client in sampled_clients:
                data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
                loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), data_loader)
                print(f"nb rounds {nb_round}: client {client}: "+ "loss {:.4f}, test accuracy {:.4f}".format(loss, acc))

    def personalize(self, nb_rounds, save_path, per_lr, rules=None, save= True):

        self.trainer.setup_lr(per_lr/10)
        self.trainer.setup_rules(rules)

        # server side
        clients = list(range(self.handler.num_clients))
        broadcast = self.handler.downlink_package
        #print(save_path)
        if save: self.save_model(save_path, model = self.handler._model, name ="global")
        
        loss, global_acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
        print("before personalization: global loss {:.4f}, test accuracy {:.4f}".format(loss, global_acc))
        
        # for p_, p in zip(self.handler._model.parameters(), self.handler.model.parameters()):
        #     print(p_, p)
        # exit()

        #client side
        global_model_data_per_client = {}
        for id, client in enumerate(clients):
            data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), data_loader)
            print(f"before personalization: client {client}: "+ "loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
            global_model_data_per_client[client] = {}
            global_model_data_per_client[client]["accuracy"] = acc
            
            if rules != None:
                rule_sat_cnt = evaluate_rules(self.handler.model, rules, data_loader)
                print(f'before personalization: client {client}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]}')
                global_model_data_per_client[client]["rules"] = [float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]
        self.trainer.personalization = True
        self.trainer.personalization_rounds = nb_rounds
        self.trainer.local_process(broadcast, clients)
        uploads = self.trainer.uplink_package

        for id, client in enumerate(clients):
            model_parameters = uploads[id][0]
            self.trainer.set_model(model_parameters)
            data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
            loss, acc = evaluate(self.trainer._model, nn.CrossEntropyLoss(), data_loader)
            print(f"after personalization (# rounds {self.trainer.personalization_rounds}): client {client}: "+ "loss {:.4f}, test accuracy {:.4f} (from {:.4f})".format(loss, acc, global_model_data_per_client[client]["accuracy"]))
            loss_g, acc_g = evaluate(self.trainer._model, nn.CrossEntropyLoss(), self.test_loader)
            print(f"after personalization: client {client}: "+ "global loss {:.4f}, test accuracy {:.4f} (from {:.4f})".format(loss_g, acc_g, global_acc))
            
            if rules != None:
                rule_sat_cnt = evaluate_rules(self.trainer._model, rules, data_loader)
                print(f'after personalization(# rounds {self.trainer.personalization_rounds}): client {client}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]} (from {global_model_data_per_client[client]["rules"] })')               
            if save: self.save_model(save_path, model = self.trainer._model, name = f"client_{client}")

    def save_model(self, path, model, name ):
        if os.path.exists(path) is not True:
            os.mkdir(path)
        torch.save(model, path+ f'/{name}.pt')

    def load_global_model(self, path):
        assert(os.path.exists(path))
        self.handler._model = torch.load(path+'/global.pt').cuda()

    def load_model(self, path):
       
        assert(os.path.exists(path)) 
        self.handler._model = torch.load(path+'/global.pt').cuda()
 

        # for p_, p in zip(self.handler._model.parameters(), self.handler.model.parameters()):
        #     print(p_, p)

        loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
        print("loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
        clients = list(range(self.handler.num_clients))
        for id, client in enumerate(clients):
            self.trainer._model = torch.load(path+f"/client_{client}.pt").cuda()
            data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
            loss, acc = evaluate(self.trainer._model, nn.CrossEntropyLoss(), data_loader)
            print(f"load personalization: client {client}: "+ "loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
    