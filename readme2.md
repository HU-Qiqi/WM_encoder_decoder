2024年7月13日 星期六 总结我的代码：
main.py  models.py  是原来的作者写的关于比特串的注入和提取器训练的代码。（我自己基于它训练出来的ckpt模型效果 目前没有HIdden文章中给出的模型文件好使）


CUDA_VISIBLE_DEVICES=1 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  bit_model/6 --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 8 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=0 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  bit_model/6 --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 8 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=3 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  bit_model/6 --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 8 --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=3 python main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 12 --epochs 350 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1


CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=8  main7.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  ckpts --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 12 --epochs 350 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --local_rank -1 --dist True --workers 2



torchrun --nproc_per_node=8 main.py \
  --val_dir path/to/coco/test2014/ --train_dir path/to/coco/train2014/ --output_dir output --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 


10.0  1   0.04这种lambda设置好像不错。
lr设置为0.0004好像不错


CUDA_VISIBLE_DEVICES=2 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1

## 2024-10-31
### main7.py model7.py  可以达到有loss_w, loss_i, loss_latents
CUDA_VISIBLE_DEVICES=7 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1 --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1


#### 后续可以看下， 分步训练

## 还可以看一下如何修改loss3

这个应该效果还好：考虑到了（1-loss3）
CUDA_VISIBLE_DEVICES=7 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1 --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1


      CUDA_VISIBLE_DEVICES=6  python  main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  ckpts/3 --eval_freq 5   --img_size 256 --num_bits 48  --batch_size 16 --epochs 361   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5     --optimizer Adam,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False  --local_rank 0



CUDA_VISIBLE_DEVICES=5  python  test.py   --test_dir datasets/Test_specific_WikiArt3   --output_dir  ckpts/1    --img_size 512 --num_bits 48   --batch_size_eval 32 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0  


## 水印
100101101011101100100110010100001101011011100110

## 2024-11-23
正在跑
CUDA_VISIBLE_DEVICES=5  python  test.py   --test_dir datasets/Test_specific_WikiArt3   --output_dir  ckpts/1    --img_size 512 --num_bits 48   --batch_size_eval 64 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0  
（进行imgs, imgs_w， imgs_generated, imgs_w_generated的生成）

之后需要做fid的计算（imgs_generated，imgs_w_generated之间的 ）


我想继续优化原来的checkpoint文件。




python main_new.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test  --test_dir datasets/big_WikiArt3_train \
  --output_dir ckpts/new  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 8 --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False  --local_rank 0





CUDA_VISIBLE_DEVICES=1,2  python -m torch.distributed.launch --nproc-per-node=2  \
  test_bit.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test  --test_dir datasets/big_WikiArt3_train \
  --output_dir ckpts/new_copy_2  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist True 


## 测试原来的ckpt(300回合)对新测试数据集的适应能力（50000）
CUDA_VISIBLE_DEVICES=1  python test_bit.py \
  --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test  --test_dir datasets/big_WikiArt3_train \
  --output_dir ckpts/new_copy_2  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False


## 

      CUDA_VISIBLE_DEVICES=6  python  main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_train --output_dir  ckpts/new_copy --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 16 --epochs 400   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5     --optimizer Adam,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False  --local_rank 0


      CUDA_VISIBLE_DEVICES=2 python main7.py   --val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1 

  
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc-per-node=4  main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist True  --local_rank 0  --master_port -1



CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node=4 main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist True  



CUDA_VISIBLE_DEVICES=1 python main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/1_new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False 

首先用 0.001 的学习率， lambda_w 10， lambda_latents_b 1
然后用 0.0005的学习率，  lambda_w 10， lambda_latents_b 2
用 0.0001， lambda_w 10, lambda_latents_b 2 (331开始的)

0.00005 336开始


## 测试
CUDA_VISIBLE_DEVICES=5  python  test.py   --test_dir datasets/Test_small_WikiArt3   --output_dir  ckpts/2    --img_size 512 --num_bits 48   --batch_size_eval 64 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0

CUDA_VISIBLE_DEVICES=1  python test_bit.py \
  --test_dir datasets/Test_small_WikiArt3 \
  --output_dir ckpts/2  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False


CUDA_VISIBLE_DEVICES=6  python  test.py   --test_dir datasets/Test_small_WikiArt3   --output_dir  ckpts/0    --img_size 512 --num_bits 48   --batch_size_eval 64 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --dist False  --local_rank 0

CUDA_VISIBLE_DEVICES=1  python test_bit.py \
  --test_dir datasets/Test_small_WikiArt3 \
  --output_dir ckpts/0  --eval_freq 5 \
  --img_size 512 --num_bits 48  --batch_size 16  --epochs 400 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist False


CUDA_VISIBLE_DEVICES=1  python  main7.py \
--val_dir datasets/big_WikiArt3_val   --train_dir datasets/big_WikiArt3_test --output_dir  ckpts/new --eval_freq 5   --img_size 512 --num_bits 48  --batch_size 12 --epochs 500   --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2   --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0   --scaling_w 0.3 --scale_channels False --attenuation none   --loss_w_type bce --loss_margin 1  --dist False


## 2024-12-4
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py \
  --val_dir datasets/coco_dataset/test2014 --train_dir datasets/coco_dataset/train2014  --output_dir output  --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --dist True --local_rank 0


CUDA_VISIBLE_DEVICES=0 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/10
--output_dir watermark/small_WikiArt3_val/10  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False



CUDA_VISIBLE_DEVICES=2 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/20
--output_dir watermark/small_WikiArt3_val/20  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False


CUDA_VISIBLE_DEVICES=3 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/30
--output_dir watermark/small_WikiArt3_val/30  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False


CUDA_VISIBLE_DEVICES=4 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/40
--output_dir watermark/small_WikiArt3_val/40  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False

CUDA_VISIBLE_DEVICES=5 python test_bit.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/50
--output_dir watermark/small_WikiArt3_val/50  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 16 --epochs 400
--scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer Lamb,lr=2e-2
--p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0
--scaling_w 0.3 --scale_channels False --attenuation none
--loss_w_type bce --loss_margin 1
--dist False

## 2024-12-30
CUDA_VISIBLE_DEVICES=3 python test_bit2.py
--test_dir /hhd2/hqq/stable_signature/Anti-DreamBooth/outputs/small_WikiArt3_val/noise-ckpt/50
--output_dir watermark/small_WikiArt3_val/50_2  --eval_freq 5
--img_size 512 --num_bits 48 --batch_size 8 --epochs 400
