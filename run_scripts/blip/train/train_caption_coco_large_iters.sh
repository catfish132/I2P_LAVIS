各个任务的训练命令

original python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip/coco_cap_ft_iter.yaml
i2p python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip/i2p_cap_ft_iter.yaml
room python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip/room_cap_ft_iter.yaml
minicoco python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip/minicoco_cap_ft_iter.yaml
