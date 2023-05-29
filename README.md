# VFL

Install the main framework

```
pip3 install fedlab
```


To train:

```
cd src/fedlab/

python3 train.py  --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 3 --lr 0.1 --major_classes_num 5 --personalization_steps 25
```

To personalize models:

```
cd src/fedlab/

python3 personalize.py  --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 3 --lr 0.1 --major_classes_num 5 --personalization_steps 25 --personalization_sim_weight 0.005

python3 personalize.py --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 2 --personalization_steps 25 --personalization_steps_replay 50 --personalization_sim_weight 0.005

python3 personalize.py --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 3 --personalization_steps 25 --personalization_steps_replay 50 --personalization_sim_weight 0.005

```

The latest model (04/05/2023):


2 major classes per client
```
python3 personalize.py --total_client 10 --com_round 15 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 2 --personalization_steps 25 --personalization_steps_replay 50 --personalization_sim_weight 0.005 --augement_data_percent_per_class  0.01

```

1 major class per client

```
python3 personalize.py --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 1 --personalization_steps 25 --personalization_steps_replay 50 --personalization_sim_weight 0.005 --augement_data_percent_per_class  0.01

```

<!-- The latest model (04/10/2023):

```
python3 personalize.py --personalization_steps_replay 50 --personalization_sim_weight 0.005   --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 1 --personalization_steps 25 --augement_data_percent_per_class  0.005 --model tinymlp --augement_data_with_zeros 250;

python3 personalize.py --personalization_steps_replay 50 --personalization_sim_weight 0.005 --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 1 --personalization_steps 25 --augement_data_percent_per_class  0.005 --model smallmlp --augement_data_with_zeros 250;
 -->

The latest model (04/10/2023) using configs for tiny ans small models, respectively:

```
python3 personalize.py --config_info ../../datasets/mnist/exps_shortcuts/shortcut_april_10_tiny.json --personalization_steps_replay 50 --personalization_sim_weight 0.005

python3 personalize.py --config_info ../../datasets/mnist/exps_shortcuts/shortcut_april_10_small.json --personalization_steps_replay 50 --personalization_sim_weight 0.005

```

# Training


To train a nano model, please perform a sequence of steps:

1. to run training
```
python3  train.py  --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 1 --lr 0.1 --major_classes_num 1 --personalization_steps 25 --augement_data_percent_per_class  0.001 --model nanomlp
```

2. after training, we will have a summary config file: "../../datasets/mnist/exps_shortcuts/config_<DATE>.json", where DATE is the current date. 

3. to run personalization

```
python3 personalize.py --config_info  ../../datasets/mnist/exps_shortcuts/config_<DATE>.json --personalization_steps_replay 50 --personalization_sim_weight 0.005 --concept_representation <decision_tree/linear> --concept_epochs 50
```



Some extra stuff (to remove later)
```
python3 train.py --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 4 --lr 0.1 --major_classes_num 1 --personalization_steps 25 --augement_data_percent_per_class  0.001 --model micromlp
python3 train.py --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 4 --lr 0.1 --major_classes_num 1 --personalization_steps 15 --augement_data_percent_per_class  0.02 --model micromlp
python3 train.py --total_client 10 --com_round 25 --sample_ratio 0.9 --batch_size 256 --epochs 4 --lr 0.1 --major_classes_num 1 --personalization_steps 15 --augement_data_percent_per_class  0.03 --model micromlp


 python3 personalize.py --config_info  ../../datasets/mnist/exps_shortcuts/config_05-16-2023-23-37-23.json --personalization_steps_replay 25 --personalization_sim_weight 5  --concept_epochs 50 --active_layers=2  --active_concepts=Curvature,Loop,'Vertical Line','Horizontal Line',Curvature,Loop,'Vertical Line','Horizontal Line' --concept_representation linear

```
