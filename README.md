# VFL
A well-known challenge of personalized federated learning is the phenomenon of *catastrophic forgetting* where fine-tuning on an individual client’s data causes the model to overfit, decreasing performance on global data. This repository includes to avoid catastrophic forgetting by using the notion of human-understandable *concepts*. We first train a global model in standard federated manner by leveraging the [FedLab](https://github.com/SMILELab-FL/FedLab) framework. Then, we extract concept representations from the global model using a dataset with concept labels. Finally, the global model is personalized for each client while ensuring that the concept representations change as little as possible. This is achieved by adding a new type of regularization term during personalization of the global model. 

The repository currently supports concept representations in the form of decision trees and linear or non-linear neural networks. Datasets currently supported are MNIST and the [Caltech-UCSD Birds](https://www.vision.caltech.edu/datasets/cub_200_2011/)(`CUB`) dataset. Both these datasets come with concept annotations. For more details about the annotations refer to [this](https://arxiv.org/abs/2209.11222) paper.


## Pre-requisites
The repository uses PyTorch, `fedlab`, `scikit-learn`, and `imblearn` packages. After installing PyTorch, setup the remaining packages as follows:
```
pip3 install fedlab imblearn
```

To use the `CUB` dataset, follow the steps below:
1. Create a folder [datasets/cub](datasets/cub) in the root folder of this repository.

2. Download the [CUB dataset](https://worksheets.codalab.org/bundles/0xd013a7ba2e88481bbc07e787f73109f5), 
and the [processed CUB dataset](https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683).

3. Extract the two downloaded archives in the created folder [datasets/cub](datasets/cub)

## Running the code
To train the global model using federated learning, run the following commands (starting from the root folder of this repository):

```
cd src/fedlab/

python3 train.py --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 4 --lr 0.1 --major_classes_num 1 --personalization_steps 25 --augement_data_percent_per_class  0.001 --dataset mnist --model micromlp
```

After training, there will be a summary config file at `../../datasets/mnist/exps_shortcuts/config_<DATE>.json`, where DATE is the current date.

To personalize the global model, run the following commands (starting from the root folder of this repository):

```
cd src/fedlab/

python3 personalize.py --config_info  ../../datasets/mnist/exps_shortcuts/config_<DATE>.json --personalization_steps_replay 25 --personalization_sim_weight 5  --concept_epochs 50 --active_layers=2  --active_concepts=Curvature,Loop,'Vertical Line','Horizontal Line' --concept_representation linear
```

The details about the options passed to `train.py` and `personalize.py` are available in the file `setup.py`.