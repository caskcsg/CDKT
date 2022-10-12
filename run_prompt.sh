python -m torch.distributed.launch --nproc_per_node=1 --use_env Hateful_prompt.py \
--config ./configs/Hateful_prompt.yaml \
--output_dir output/Hateful_prompt \
--checkpoint output/Hateful/ALBEF.pth \
--dist_url tcp://127.0.0.1:50005
