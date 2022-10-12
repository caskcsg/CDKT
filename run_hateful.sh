python -m torch.distributed.launch --nproc_per_node=1 --use_env Hateful.py \
--config ./configs/Hateful.yaml \
--output_dir output/Hateful_4 \
--checkpoint output/Hateful/ALBEF.pth \
--dist_url tcp://127.0.0.1:50009

