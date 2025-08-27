
CUDA_VISIBLE_DEVICES=0 python eval_model.py checkpoint --dataset /data/DSCVC/Dataset/SCC-SEQ -exp ckpts/dscvc_ckpt.pth.tar --output ../result-all -est --cuda --frames -1 -ri  0 1 2 3 --gop 8 --gpu_id 0 --save
