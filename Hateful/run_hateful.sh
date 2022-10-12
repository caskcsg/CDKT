CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --use_env Hateful.py \
--config ../configs/Hateful.yaml \
--output_dir save/16 \
--checkpoint ../output/Hateful/ALBEF.pth \
--dist_url tcp://127.0.0.1:50003


