# VFL

Install the main framework

```
pip3 install fedlab
```


To train:

```
cd src/fedlab/

python3 standalone.py  --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 3 --lr 0.1 --major_classes_num 5 --personalization_steps 25
```

To load models:

```
cd src/fedlab/

python3 load_models.py  --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 3 --lr 0.1 --major_classes_num 5 --personalization_steps 25 --personalization_sim_weight 0.005

python3 load_models.py  --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 2 --personalization_steps 25 --personalization_steps_replay 50 --personalization_sim_weight 0.005

python3 load_models.py  --total_client 10 --com_round 50 --sample_ratio 0.9 --batch_size 256 --epochs 5 --lr 0.1 --major_classes_num 3 --personalization_steps 25 --personalization_steps_replay 50 --personalization_sim_weight 0.005

```
