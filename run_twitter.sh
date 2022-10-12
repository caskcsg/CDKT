python -m torch.distributed.launch --nproc_per_node=1 --use_env Twitter.py \
--config ./configs/Twitter.yaml \
--output_dir output/Twitter_metircs \
--checkpoint output/Hateful/ALBEF.pth
