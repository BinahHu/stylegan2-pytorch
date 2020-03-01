python  -m torch.distributed.launch --nproc_per_node=4 --master_port=233666 \
        train.py /home/huzy/datasets/COCO/ \
        --iter 3000 --n_sample 16 --batch 4