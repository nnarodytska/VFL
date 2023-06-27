import torch
import torch.nn as nn
from utils import get_device

def get_mnist_model(args):
    device = get_device(args.cuda, args.device)

    if args.model == "real":
        return ClassifierMnist(784).to(device)

    if args.model == "mlp":
        return MLP(784, 10).to(device)

    if args.model == "smallmlp":
        return SmallMLP(784, 10).to(device)

    if args.model == "tinymlp":
        return TinyMLP(784, 10).to(device)
    
    if args.model == "micromlp":
        if  not hasattr(args, 'active_layers') or  args.active_layers is None:
            return MicroMLP(784, 10).to(device)
        else:
            return MicroMLP(784, 10, active_layers=args.active_layers).to(device)

    if args.model == "nanomlp":
        return NanoMLP(784, 10).to(device)

class LinearLayerConcept(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super(LinearLayerConcept, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  
     

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class SmallLayerConcept(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super(SmallLayerConcept, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, 10)  
        self.fc2 = nn.Linear(10, 10)  
        self.fc3 = nn.Linear(10, output_dim)  
     

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)        
        x = self.relu(x)
        x = self.fc3(x)   
        return x

class ClassifierMnist(nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(ClassifierMnist, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.checkpoints_files = []
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.cnn2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 2 * self.latent_dim)
        self.fc2 = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

    def input_to_representation(self, x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def representation_to_output(self, h):
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.out(h)
        return h


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def input_to_representation(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class SmallMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()

    def input_to_representation(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class TinyMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(TinyMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()
        
    def input_to_representation(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MicroMLP(nn.Module):
    def __init__(self, input_size, output_size, active_layers = None):
        super(MicroMLP, self).__init__()
        nb_hidden_1 = 20
        nb_hidden_2 = 10
        self.fc1 = nn.Linear(input_size, nb_hidden_1)
        self.fc2 = nn.Linear(nb_hidden_1, nb_hidden_2)
        self.fc3 = nn.Linear(nb_hidden_2, output_size)
        self.relu = nn.ReLU()
        self.probe_mode = False


        self.concepts = ["Curvature", "Loop", "Vertical Line", "Horizontal Line", "Curvature", "Loop", "Vertical Line", "Horizontal Line"]
        # self.curvature_probe_h1 = nn.Linear(nb_hidden_1, 2, bias=True)
        # self.loop_probe_h1 = nn.Linear(nb_hidden_1, 2, bias=True)
        # self.vline_probe_h1 = nn.Linear(nb_hidden_1, 2, bias=True)
        # self.hline_probe_h1 = nn.Linear(nb_hidden_1, 2, bias=True)

        self.curvature_probe_h2 = SmallLayerConcept(nb_hidden_2, 2, bias=True)
        self.loop_probe_h2  = SmallLayerConcept(nb_hidden_2, 2, bias=True)
        self.vline_probe_h2 =SmallLayerConcept(nb_hidden_2, 2, bias=True)
        self.hline_probe_h2 =SmallLayerConcept(nb_hidden_2, 2, bias=True)


        self.concept_layers = [self.curvature_probe_h2, self.loop_probe_h2, self.vline_probe_h2, self.hline_probe_h2]

        self.all_layers = [self.fc1, self.fc2, self.fc3]

        self.pred_layers = self.all_layers
        if not(active_layers is None):   
            self.pred_layers =  [ self.all_layers[i] for i in active_layers]
        print( f"self.pred_layers {self.pred_layers}")


    def start_probe_mode(self):
        self.probe_mode = True

    def stop_probe_mode(self):
        self.probe_mode = False

    def input_to_representation(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x



    def forward(self, x):
        concept_outputs = []
       
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
    
        ############
        x = self.fc2(x)       
        concept_outputs.append(self.curvature_probe_h2(x))
        concept_outputs.append(self.loop_probe_h2(x))
        concept_outputs.append(self.vline_probe_h2(x))
        concept_outputs.append(self.hline_probe_h2(x))
        x = self.relu(x)

        ##############
        x = self.fc3(x)

        #print(f" self.probe_mode {self.probe_mode}")
        if self.probe_mode:
            return x, *concept_outputs
        else:
            return x

    def probe(self, x):
        x = x.view(x.shape[0], -1)
        ###########
        x = self.fc1(x)
        output = []
        x = self.relu(x)

        ##############        
        x = self.fc2(x)
        for concept_layer in self.concept_layers[4:]:
            output.append(concept_layer(x))
        
        
        return tuple(output)

class NanoMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(NanoMLP, self).__init__()
        self.probe_mode = False
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()
        
        self.concepts = ["Curvature", "Loop", "Vertical Line", "Horizontal Line"]
        self.curvature_probe = nn.Linear(20, 2, bias=False)
        self.loop_probe = nn.Linear(20, 2, bias=False)
        self.vline_probe = nn.Linear(20, 2, bias=False)
        self.hline_probe = nn.Linear(20, 2, bias=False)

        self.concept_layers = [self.curvature_probe, self.loop_probe, self.vline_probe, self.hline_probe]
        self.pred_layers = [self.fc1, self.fc2]

    def start_probe_mode(self):
        self.probe_mode = True

    def stop_probe_mode(self):
        self.probe_mode = False

    def input_to_representation(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        concept_outputs = []
        concept_outputs.append(self.curvature_probe(x))
        concept_outputs.append(self.loop_probe(x))
        concept_outputs.append(self.vline_probe(x))
        concept_outputs.append(self.hline_probe(x))
        x = self.fc2(x)
        if self.probe_mode:
            return x, *concept_outputs
        else:
            return x
    
    def probe(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        output = []
        for concept_layer in self.concept_layers:
            output.append(concept_layer(x))
        return tuple(output)

