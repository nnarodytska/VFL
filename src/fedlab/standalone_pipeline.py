import os
from fedlab.core.standalone import StandalonePipeline
import torch
from torch import nn
from utils import evaluate_rules, evaluate_label_specific, plot_client_stats, evaluate_linear_concepts, evaluate
import collections.abc

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


    def personalize(self, nb_rounds, save_path, per_lr, rules=None, sim_weight = 1, save= True, debug = 0):
        def print_statistics(stats):
            for k,v in stats.items():
                if isinstance(v, collections.abc.Sequence):
                    print(f"\t{k: <45}: {[round(w,2) for w in v]}")
                else:
                    print(f"\t{k: <45}: {[round(v,2)]}")
        
        self.trainer.setup_lr(per_lr/10)
        self.trainer.setup_rules(rules)
        self.trainer.setup_sim_weight(sim_weight)
        self.trainer.setup_global_model(self.handler._model)

        # server side
        clients = list(range(self.handler.num_clients))
        broadcast = self.handler.downlink_package
        #print(save_path)
        if save: self.save_model(save_path, model = self.handler._model, name ="global")
        
        loss, global_acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
        # print("before personalization: global loss {:.4f}, test accuracy {:.4f}".format(loss, global_acc))
        
        # for p_, p in zip(self.handler._model.parameters(), self.handler.model.parameters()):
        #     print(p_, p)
        # exit()

        #client side
        client_stats_pre_personalization = {}
        client_stats_post_personalization = {}
        for id, client in enumerate(clients):
            data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), data_loader)
            if debug == 1: print(f"before personalization: client {client}: "+ "loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
            label_specific_acc_local = evaluate_label_specific(self.handler.model, data_loader)
            label_specific_acc_global = evaluate_label_specific(self.handler.model, self.test_loader)

            client_stats_pre_personalization[client] = {}
            client_stats_pre_personalization[client]["global_accuracy"] = global_acc
            client_stats_pre_personalization[client]["local_accuracy"] = acc
            client_stats_pre_personalization[client]["label_specific_local_accuracy"] = label_specific_acc_local
            client_stats_pre_personalization[client]["label_specific_global_accuracy"] = label_specific_acc_global
            if rules != None:
                rule_sat_cnt = evaluate_rules(self.handler.model, rules, data_loader)
                # print(f'before personalization: client {client}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]}')
                client_stats_pre_personalization[client]["rules_local"] = [float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]
                
                rule_sat_cnt = evaluate_rules(self.handler.model, rules, self.test_loader)
                # print(f'before personalization: client {client}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]}')
                client_stats_pre_personalization[client]["rules_global"] = [float(cnt) / len(self.test_loader.dataset) for cnt in rule_sat_cnt]
            
            if self.trainer.concept_representation == "linear":
                concept_present_count = evaluate_linear_concepts(self.handler.model, data_loader)
                client_stats_pre_personalization[client]["concepts_local"] = [float(cnt) / len(data_loader.dataset) for cnt in concept_present_count]
                concept_present_count = evaluate_linear_concepts(self.handler.model, self.test_loader)
                client_stats_pre_personalization[client]["concepts_global"] = [float(cnt) / len(self.test_loader.dataset) for cnt in concept_present_count]

        original_epoch = self.trainer.epochs
        self.trainer.epochs  = 100
        self.trainer.personalization = True
        self.trainer.personalization_rounds = nb_rounds
        self.trainer.local_process(broadcast, clients)
        uploads = self.trainer.uplink_package

        self.trainer.epochs  = original_epoch

        for id, client in enumerate(clients):
            model_parameters = uploads[id][0]
            self.trainer.set_model(model_parameters)
            data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
            loss, acc = evaluate(self.trainer._model, nn.CrossEntropyLoss(), data_loader)
            if debug == 1: print(f"after personalization (# rounds {self.trainer.personalization_rounds}): \
                  client {client}: "+ "loss {:.4f}, test accuracy {:.4f} (from {:.4f})".format(loss, acc, client_stats_pre_personalization[client]["local_accuracy"]))
            loss_g, acc_g = evaluate(self.trainer._model, nn.CrossEntropyLoss(), self.test_loader)
            if debug == 1: print(f"after personalization: client {client}: "+ "global loss {:.4f}, test accuracy {:.4f} (from {:.4f})".format(loss_g, acc_g, client_stats_pre_personalization[client]["global_accuracy"]))
            label_specific_acc_local = evaluate_label_specific(self.trainer._model, data_loader)
            label_specific_acc_global = evaluate_label_specific(self.trainer._model, self.test_loader)

            client_stats_post_personalization[client] = {}
            client_stats_post_personalization[client]["global_accuracy"] = acc_g
            client_stats_post_personalization[client]["local_accuracy"] = acc
            client_stats_post_personalization[client]["label_specific_local_accuracy"] = label_specific_acc_local
            client_stats_post_personalization[client]["label_specific_global_accuracy"] = label_specific_acc_global

            if rules != None:
                rule_sat_cnt = evaluate_rules(self.trainer._model, rules, data_loader)
                # print(f'after personalization(# rounds {self.trainer.personalization_rounds}): \
                #       client {client}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]} (from {client_stats_pre_personalization[client]["rules"] })')
                client_stats_post_personalization[client]["rules_local"] = [float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]

                rule_sat_cnt = evaluate_rules(self.trainer._model, rules, self.test_loader)
                # print(f'after personalization(# rounds {self.trainer.personalization_rounds}): \
                #       client {client}: % of inputs satisfying rules {[float(cnt) / len(data_loader.dataset) for cnt in rule_sat_cnt]} (from {client_stats_pre_personalization[client]["rules"] })')
                client_stats_post_personalization[client]["rules_global"] = [float(cnt) / len(self.test_loader.dataset) for cnt in rule_sat_cnt]
            
            if self.trainer.concept_representation == "linear":
                concept_present_count = evaluate_linear_concepts(self.trainer._model, data_loader)
                client_stats_post_personalization[client]["concepts_local"] = [float(cnt) / len(data_loader.dataset) for cnt in concept_present_count]
                concept_present_count = evaluate_linear_concepts(self.trainer._model, self.test_loader)
                client_stats_post_personalization[client]["concepts_global"] = [float(cnt) / len(self.test_loader.dataset) for cnt in concept_present_count]
      
            if save: self.save_model(save_path, model = self.trainer._model, name = f"client_{client}")
            

        print("\nBefore personalization results:")
        for id, client in enumerate(clients):
            print(f'Client {client:<2}: ')
            print_statistics(client_stats_pre_personalization[client])
            plot_client_stats(client_stats_pre_personalization[client], id, "pre")
        print("\nAfter personalization results:")
        for id, client in enumerate(clients):
            print(f'Client {client:<2}: ')
            print_statistics(client_stats_post_personalization[client])
            plot_client_stats(client_stats_post_personalization[client], id, "post")


    def save_model(self, path, model, name ):
        if os.path.exists(path) is not True:
            os.mkdir(path)
        torch.save(model, path+ f'/{name}.pt')

    def load_global_model(self, path):
        #print(path)
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
            print(self.trainer._model)
            data_loader = self.trainer.dataset.get_dataloader(client, self.trainer.batch_size)
            loss, acc = evaluate(self.trainer._model, nn.CrossEntropyLoss(), data_loader)
            print(f"load personalization: client {client}: "+ "loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
    