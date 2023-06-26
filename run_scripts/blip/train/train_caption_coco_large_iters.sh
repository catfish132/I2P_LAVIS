original python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip/coco_cap_ft_iter.yaml
i2p python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip/i2p_cap_ft_iter.yaml
