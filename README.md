# VFL

To train:

```
python3 standalone.py  --total_client 10 --com_round 2 --sample_ratio 0.9 --batch_size 256 --epochs 3 --lr 0.1 --major_classes_num 5 --personalization_steps 5
```

To laod models:

```
python3 load_models.py  --total_client 10 --com_round 2 --sample_ratio 0.9 --batch_size 256 --epochs 3 --lr 0.1 --major_classes_num 5 --personalization_steps 5
```