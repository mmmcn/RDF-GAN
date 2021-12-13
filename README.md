# Inference

## NYUDepthV2

1. $T^* \Rightarrow T$ :

   ```shell
   bash ./bash/test_nyuv2_s2d.sh --dataset nyudepthv2_s2d --data_root data/nyudepthv2 --load_from ${CHECKPOINT_DIR} --work_dir ${WORK_DIR} --batch_size 1
   ```

2. $R^* \Rightarrow T$ :

   ```shell
   bash ./bash/test_nyuv2_s2d.sh --dataset nyuv21400_s2d --data_root data/nyuv2 --load_from ${CHECKPOINT_DIR} --work_dir ${WORK_DIR} --batch_size 1
   ```

3. $R \Rightarrow T$ :

   ```shell
   bash ./bash/test_nyuv2_r2r.sh  --work_dir ${WORK_DIR} --load_from ${CHECKPOINT_DIR} --batch_size 1
   ```

## SUN RGBD

1. $R \Rightarrow T$ :

   ```shell
   bash ./bash/test_sunrgbd_r2r.sh --work_dir ${WORK_DIR} --load_from ${CHECKPOINT_DIR} --batch_size 1
   
   bash ./bash/vanilla/test_sunrgbd_r2r.sh --work_dir ${WORK_DIR} --load_from ${CHECKPOINT_DIR} --batch_size 1
   ```

   