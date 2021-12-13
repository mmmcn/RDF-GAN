import os
from config_vanilla import args
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))

import torch
import torch.nn as nn
from torch import distributed as dist
from lib.utils import set_random_seed
import torch.nn.functional as F
import torch.optim as optim
from lib.tools.helper import (Logger, MovingAverage, reduce_loss,
                              save_checkpoint, init_weights)
from lib.tools.helper import resume_from_vanilla as resume_from
from lib.tools.helper import load_checkpoint_vanilla as load_checkpoint
from torch.autograd import grad
from lib.evaluator.rdf_gan_evaluator import (Eval, DistEval)
import time
import matplotlib.pyplot as plt


def build_model(args):
    from lib.models.segmentator.esa_net.esa_net_one_modality import ESANetOneModality
    from lib.models.generator.rdf_gan_generator.rdf_gan_generator_vanilla import DCVGANGenerator
    from lib.models.discriminator.patch_gan_discriminator import PatchGANDiscriminator

    # build the global_guidance_module, use U-net or ESANet-One-Modality
    global_guidance_module_out_channels_0 = args.glo_guid_channels_out_0
    global_guidance_module_out_channels_1 = args.glo_guid_channels_out_1
    encoder_decoder_module = ESANetOneModality(height=args.height,
                                               width=args.width,
                                               num_classes=args.num_classes,
                                               pretrained_on_imagenet=args.pretrained_on_imagenet,
                                               pretrained_dir=args.guidance_pretrained_dir,
                                               encoder=args.guidance_encoder,
                                               encoder_block=args.guidance_encoder_block,
                                               channels_decoder=args.guidance_channels_decoder,
                                               nr_decoder_blocks=args.guidance_nr_decoder_blocks,
                                               encoder_decoder_fusion=args.guidance_encoder_decoder_fusion,
                                               context_module=args.guidance_context_module,
                                               weighting_in_encoder=args.guidance_weighting_in_encoder,
                                               upsampling=args.guidance_upsampling,
                                               pyramid_supervision=False)
    # if args.use_pretrained_encoder_decoder:
    #     from lib.tools.train_segmentator import load_checkpoint as load_segmentator_checkpoint
    #     checkpoint = load_segmentator_checkpoint(encoder_decoder_module,
    #                                              args.load_encoder_decoder_path,
    #                                              map_location=torch.device('cpu'))


    from collections import OrderedDict
    if not args.separate_global_guidance_module:
        global_guidance_module = nn.Sequential(OrderedDict([
            ('encoder_decoder', encoder_decoder_module),
            ('identity', nn.Identity()) if not args.reduction_glo_guid_module else\
            ('conv2d', nn.Conv2d(args.num_classes, global_guidance_module_out_channels_1, kernel_size=3, padding=1))
        ]))

    else:
        global_guidance_module = nn.ModuleList()
        global_guidance_module.append(encoder_decoder_module)
        global_guidance_module.append(nn.Identity() if not args.reduction_glo_guid_module else\
                                      nn.Conv2d(args.num_classes,global_guidance_module_out_channels_1,
                                                kernel_size=3, padding=1))

    # global_guidance_module = nn.Sequential(OrderedDict([
    #     ('encoder_decoder', nn.Conv2d(3, args.num_classes, kernel_size=3, padding=1)),
    #     ('identity', nn.Identity()) if not args.reduction_glo_guid_module else \
    #         ('conv2d', nn.Conv2d(args.num_classes, global_guidance_module_out_channels, kernel_size=3, padding=1))
    # ]))


    generator = DCVGANGenerator(global_guidance_module=global_guidance_module,
                                global_guidance_module_out_channels_0=global_guidance_module_out_channels_0,
                                global_guidance_module_out_channels_1=global_guidance_module_out_channels_1,
                                encoder_rgb=args.encoder_rgb,
                                encoder_depth=args.encoder_depth,
                                encoder_block=args.encoder_block,
                                rgb_channels_decoder=args.rgb_channels_decoder,
                                depth_channels_decoder=args.depth_channels_decoder,
                                nr_decoder_blocks=args.nr_decoder_blocks,
                                pretrained_on_imagenet=False,
                                fuse_depth_in_rgb_encoder=None if args.fuse_depth_in_rgb_encoder == 'None' else args.fuse_depth_in_rgb_encoder,
                                fuse_depth_in_rgb_decoder=None if args.fuse_depth_in_rgb_decoder == 'None' else args.fuse_depth_in_rgb_decoder,
                                encoder_decoder_fusion=args.encoder_decoder_fusion,
                                activation=args.activation,
                                norm_layer_type=args.norm_layer_type,
                                upsampling_mode=args.upsampling_mode,
                                adain_weighting=True if args.adain_weighting else False,
                                separate_global_guidance_module=True if args.separate_global_guidance_module else False,
                                use_pretrained_global_guidance_module=True if args.use_pretrained_encoder_decoder else False)

    discriminator_rgb = PatchGANDiscriminator(in_channels=1,
                                              norm_cfg=dict(type='BN') if args.disc_norm_type.lower() == 'bn' else dict(type='IN'),
                                              activation='LeakyReLU' if args.disc_act_type.lower() == 'leakyrelu' else 'ReLU')
    discriminator_depth = PatchGANDiscriminator(in_channels=1,
                                                norm_cfg=dict(type='BN') if args.disc_norm_type.lower() == 'bn' else dict(type='IN'),
                                                activation='LeakyReLU' if args.disc_act_type.lower() == 'leakyrelu' else 'ReLU')

    # generator's parameters are initialized inside the class

    # initializes the discriminator parameters explicitly
    if args.init_disc:
        init_weights(discriminator_rgb)
        init_weights(discriminator_depth)

    return generator, discriminator_rgb, discriminator_depth


def get_dataloader(args):
    from lib.dataset.build_dataloader import build_dataloader

    if args.dataset == 'sunrgbd':
        from lib.dataset.sunrgbd.sunrgbd_dataset import SUNRGBDPseudoDataset
        DATASET = SUNRGBDPseudoDataset

        dataset_kwargs = dict(max_depth=10.0,
                              rgb_mean=[0.485, 0.456, 0.406],
                              rgb_std=[0.229, 0.224, 0.225],
                              depth_mean=[5.0],
                              depth_std=[5.0],
                              )

    elif args.dataset == 'nyudepthv2_s2d':
        from lib.dataset.nyuv2.nyuv2_sparse_to_dense_dataset import NYUV2S2DDataset
        DATASET = NYUV2S2DDataset

        dataset_kwargs = dict(num_sample=500,
                              max_depth=10.0,
                              depth_mean=[5.0],
                              depth_std=[5.0]
                              )

    elif args.dataset == 'nyudepthv2_r2r':
        from lib.dataset.nyuv2.nyuv2_raw_to_reconstructed_dataset import NYUV2R2RDataset

        DATASET = NYUV2R2RDataset

        dataset_kwargs = dict(num_sample=500,
                              max_depth=10.0,
                              depth_mean=[5.0],
                              depth_std=[5.0]
                              )
    elif args.dataset == 'nyuv21400_s2d':
        from lib.dataset.nyuv2.nyuv2_1400_sparse_to_dense_dataset import NYUV21400S2DDataset

        DATASET = NYUV21400S2DDataset
        dataset_kwargs = dict(num_sample=500,
                              max_depth=10.0,
                              depth_mean=[5.0],
                              depth_std=[5.0]
                              )
    else:
        raise NotImplementedError(f'Only SUN RGBD, NYUDepthv2, are supported so far, '
                                  f'but got {args.dataset}')


    train_dataset = DATASET(data_root=args.data_root,
                            mode='train',
                            **dataset_kwargs)

    val_dataset = DATASET(data_root=args.data_root,
                          mode='test' if args.dataset in ['nyudepthv2_s2d'] else 'val',
                          **dataset_kwargs)

    train_dataloader = build_dataloader(train_dataset,
                                        samples_per_gpu=args.batch_size,
                                        workers_per_gpu=args.num_workers,
                                        num_gpus=len(args.gpus),
                                        dist=len(args.gpus) > 1,
                                        seed=args.seed,
                                        pin_memory=False,
                                        drop_last=True)
    val_dataloader = build_dataloader(val_dataset,
                                      samples_per_gpu=args.batch_size,
                                      workers_per_gpu=0,
                                      num_gpus=len(args.gpus),
                                      dist=len(args.gpus) > 1,
                                      pin_memory=False,
                                      shuffle=False)

    return train_dataloader, val_dataloader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main():
    EPSION = 1e-6

    args.gpus = [int(i) for i in args.gpus.split(',')]
    distributed = args.num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    logger = Logger(name='RDF-GAN', local_rank=args.local_rank, save_dir=args.work_dir)

    logger.log(str(args))

    if args.local_rank == 0:
        args.sample_dir = os.path.join(args.work_dir, 'samples')
        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)

    set_random_seed(args.seed)

    # build dataloader
    train_dataloader, val_dataloader = get_dataloader(args)
    fixed_samples = next(iter(val_dataloader))

    # build model, including generator and discriminator
    generator, discriminator_rgb, discriminator_depth = build_model(args)

    # put model to corresponding device
    device = torch.device("cuda", args.local_rank)
    if not distributed:
        generator = generator.to(device)
        discriminator_rgb = discriminator_rgb.to(device)
        discriminator_depth = discriminator_depth.to(device)
    else:
        generator = torch.nn.parallel.DistributedDataParallel(generator.to(device),
                                                              device_ids=[args.local_rank],
                                                              # find_unused_parameters=True
                                                              )
        discriminator_rgb = torch.nn.parallel.DistributedDataParallel(discriminator_rgb.to(device),
                                                                      device_ids=[args.local_rank],
                                                                      # find_unused_parameters=True
                                                                      )
        discriminator_depth = torch.nn.parallel.DistributedDataParallel(discriminator_depth.to(device),
                                                                        device_ids=[args.local_rank],
                                                                        # find_unused_parameters=True
                                                                        )

    evaluator = Eval(val_dataloader, device=device) if not distributed else DistEval(val_dataloader, device=device)

    # optimizer hyper parameters setting
    learning_rate = args.learning_rate
    beta1, beta2 = args.beta1, args.beta2

    # filter parameters that do not want to be updated here
    if args.use_pretrained_encoder_decoder:
        generator_params = [{"params": [p for n, p in generator.named_parameters() if 'global_guidance_module' not in n and p.requires_grad]},
                            {"params": [p for n, p in generator.named_parameters() if 'global_guidance_module' in n and p.requires_grad],
                             "lr": learning_rate * 0.5}]
    else:
        generator_params = generator.parameters()

    if args.optimizer.lower() == 'adam':
        optimizer_generator = optim.Adam(generator_params, lr=learning_rate, betas=(beta1, beta2))
        optimizer_disc_rgb = optim.Adam(discriminator_rgb.parameters(), lr=learning_rate, betas=(beta1, beta2))
        optimizer_disc_depth = optim.Adam(discriminator_depth.parameters(), lr=learning_rate, betas=(beta1, beta2))
    elif args.optimizer.lower() == 'sgd':
        optimizer_generator = optim.SGD(generator_params, lr=learning_rate)
        optimizer_disc_rgb = optim.SGD(discriminator_rgb.parameters(), lr=learning_rate)
        optimizer_disc_depth = optim.SGD(discriminator_depth.parameters(), lr=learning_rate)

    elif args.optimizer.lower() == 'rmsprop':
        optimizer_generator = optim.RMSprop(generator_params, lr=learning_rate)
        optimizer_disc_rgb = optim.RMSprop(discriminator_rgb.parameters(), lr=learning_rate)
        optimizer_disc_depth = optim.RMSprop(discriminator_depth.parameters(), lr=learning_rate)

    else:
        raise NotImplementedError(f'Only Adam, SGD, RMSprop optimizers are supported, but got {args.optimizer}')

    if args.lr_scheduler == 'step':
        lr_scheduler_generator = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_generator,
                                                                milestones=args.lr_decay_epochs,
                                                                gamma=args.lr_decay_rate)
        lr_scheduler_disc_rgb = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_disc_rgb,
                                                               milestones=args.lr_decay_epochs,
                                                               gamma=args.lr_decay_rate)
        lr_scheduler_disc_depth = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_disc_depth,
                                                                 milestones=args.lr_decay_epochs,
                                                                 gamma=args.lr_decay_rate)

    elif args.lr_scheduler == 'onecycle':
        lr_scheduler_generator = optim.lr_scheduler.OneCycleLR(optimizer=optimizer_generator,
                                                               max_lr=[i['lr'] for i in optimizer_generator.param_groups],
                                                               total_steps=args.max_epoch,
                                                               div_factor=args.div_factor,
                                                               pct_start=args.pct_start,
                                                               anneal_strategy='cos',
                                                               final_div_factor=1e4)
        lr_scheduler_disc_rgb = optim.lr_scheduler.OneCycleLR(optimizer=optimizer_disc_rgb,
                                                              max_lr=[i['lr'] for i in optimizer_disc_rgb.param_groups],
                                                              total_steps=args.max_epoch,
                                                              div_factor=args.div_factor,
                                                              pct_start=args.pct_start,
                                                              anneal_strategy='cos',
                                                              final_div_factor=1e4)
        lr_scheduler_disc_depth = optim.lr_scheduler.OneCycleLR(optimizer=optimizer_disc_depth,
                                                                max_lr=[i['lr'] for i in optimizer_disc_depth.param_groups],
                                                                total_steps=args.max_epoch,
                                                                div_factor=args.div_factor,
                                                                pct_start=args.pct_start,
                                                                anneal_strategy='cos',
                                                                final_div_factor=1e4)
    elif args.lr_scheduler == 'cosine':
        # FIXME: The CosineAnnealingLR used before has problems, the final learning rate will not drop to zero, it only take half a cycle.

        lr_scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_generator,
                                                                      # T_max=args.max_epoch + 10
                                                                      T_max=args.t_max
                                                                      )
        lr_scheduler_disc_rgb = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_disc_rgb,
                                                                     # T_max=args.max_epoch + 10,
                                                                     T_max=args.t_max
                                                                     )
        lr_scheduler_disc_depth = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_disc_depth,
                                                                       # T_max=args.max_epoch + 10
                                                                       T_max=args.t_max
                                                                       )
    else:
        raise NotImplementedError(f'Only multi step, cosine annealing and onecycle lr scheduler are supported'
                                  f'but got {args.lr_schdeuler}')

    start_epoch = 0
    num_iters_per_epoch = len(train_dataloader)

    if args.resume_from is not None:
        # warp the parts to dict type
        start_epoch = resume_from(models=dict(generator=generator,
                                              disc_rgb=discriminator_rgb,
                                              disc_depth=discriminator_depth),
                                  optimizers=dict(generator=optimizer_generator,
                                  disc_rgb=optimizer_disc_rgb,
                                  disc_depth=optimizer_disc_depth),
                                  lr_schedulers=dict(generator=lr_scheduler_generator,
                                                     disc_rgb=lr_scheduler_disc_rgb,
                                                     disc_depth=lr_scheduler_disc_depth),
                                  filename=args.resume_from,
                                  logger=logger)

    if args.load_from is not None:
        logger.log(f'load checkpoint from {args.load_from}')
        checkpoint = load_checkpoint(model=dict(generator=generator,
                                                disc_rgb=discriminator_rgb,
                                                disc_depth=discriminator_depth),
                                     filename=args.load_from,
                                     map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
                                     logger=logger)

    if args.cal_fps:
        num_warmup = 5
        pure_inf_time = 0
        log_interval = 50
        samples = 300

        generator.eval()
        for i, data in enumerate(val_dataloader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                generator(data['rgb'].to(device),
                          data['raw_depth'].to(device))

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {samples}], '
                          f'fps: {fps:.1f} img / s')

            if (i + 1) == samples:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.1f} img / s')
                break

        generator.train()
        exit(0)


    # import numpy as np
    # def colored_depth_map(depth, d_min=None, d_max=None, cmap=plt.cm.viridis):
    #     if d_min is None:
    #         d_min = np.min(depth)
    #     if d_max is None:
    #         d_max = np.max(depth)
    #
    #     depth_relative = (depth - d_min) / (d_max - d_min)
    #     return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
    #
    # from PIL import Image
    #
    # def save_img(arr, save_path):
    #     img = Image.fromarray(arr)
    #     img.save(save_path)


    if args.inference:
        evaluator.evaluate(generator, logger)

        # generator.eval()
        # for i, data in enumerate(val_dataloader):
        #     with torch.no_grad():
        #         depth1, conf1, depth2, conf2 = generator(data['rgb'].to(device),
        #                                                  data['raw_depth'].to(device),
        #                                                  idx=i)
        #         depth1 = torch.tanh(depth1)
        #         depth2 = torch.tanh(depth2)
        #         conf = torch.cat([conf1, conf2], dim=1)
        #         conf_score = F.softmax(conf, 1)
        #         pred_depth_map = torch.cat([depth1, depth2], dim=1)
        #         pred_depth_map = torch.sum(pred_depth_map * conf_score, dim=1, keepdim=True)
        #
        #         #------------------------ save images for sunrgbd pseudo -----------------------
        #         # pred depth value & colored depth
        #         # path_save_pred = 'vis_tmp/pred/{:04d}.png'.format(i)
        #         # path_save_pred_colored = 'vis_tmp/pred_colored/{:04d}.png'.format(i)
        #         #
        #         # pred = pred_depth_map.cpu().squeeze().detach().numpy()
        #         # pred = pred * 5.0 + 5.0
        #         # pred_colored = colored_depth_map(pred)
        #         # pred = (pred * 25.5).astype(np.uint8)
        #
        #         #
        #         # save_img(pred, path_save_pred)
        #         # save_img(pred_colored.astype(np.uint8), path_save_pred_colored)
        #         # ------------------------ save images for sunrgbd pseudo -----------------------
        #
        #
        #         # ------------------------ save images for difference branch -----------------------
        #         # pred depth value & colored depth
        #         path_save_pred = 'vis_tmp/part/pred/{:04d}.png'.format(i)
        #         path_save_pred_colored = 'vis_tmp/part/pred_colored/{:04d}.png'.format(i)
        #         pred = pred_depth_map.cpu().squeeze().detach().numpy()
        #         pred = pred * 5.0 + 5.0
        #         pred_colored = colored_depth_map(pred)
        #         pred = (pred * 25.5).astype(np.uint8)
        #         save_img(pred, path_save_pred)
        #         save_img(pred_colored.astype(np.uint8), path_save_pred_colored)
        #
        #         path_save_rgb_branch = 'vis_tmp/part/rgb_branch/{:04d}.png'.format(i)
        #         path_save_rgb_branch_colored = 'vis_tmp/part/rgb_branch_colored/{:04d}.png'.format(i)
        #         rgb = depth1.cpu().squeeze().detach().numpy()
        #         rgb = rgb * 5.0 + 5.0
        #         rgb_colored = colored_depth_map(rgb)
        #         rgb = (rgb * 25.5).astype(np.uint8)
        #         save_img(rgb, path_save_rgb_branch)
        #         save_img(rgb_colored.astype(np.uint8), path_save_rgb_branch_colored)
        #
        #         path_save_depth_branch = 'vis_tmp/part/depth_branch/{:04d}.png'.format(i)
        #         path_save_depth_branch_colored = 'vis_tmp/part/depth_branch_colored/{:04d}.png'.format(i)
        #         depth = depth2.cpu().squeeze().detach().numpy()
        #         depth = depth * 5.0 + 5.0
        #         depth_colored = colored_depth_map(depth)
        #         depth = (depth * 25.5).astype(np.uint8)
        #         save_img(depth, path_save_depth_branch)
        #         save_img(depth_colored.astype(np.uint8), path_save_depth_branch_colored)
        #         # ------------------------ save images for difference branch -----------------------
        #
        #
        #
        #         # --------------save array for visiualization-------------- #
        #         # save_pickle(tensor=depth1, idx=i, prefix='rgb_branch_depth')
        #         # save_pickle(tensor=depth2, idx=i, prefix='depth_branch_depth')
        #         # save_pickle(tensor=pred_depth_map, idx=i, prefix='final_depth')
        #         # save_pickle(tensor=conf_score, idx=i, prefix='conf_score')
        #         # save_pickle(tensor=data['gt_depth'], idx=i, prefix='gt_depth')
        #         # --------------save array for visiualization-------------- #
        #
        #     print(f'\r{i * val_dataloader.batch_size}/{len(val_dataloader)}', end='')
        #
        #     # sample = dict(rgb=data['rgb'],
        #     #               raw_depth=data['raw_depth'],
        #     #               pred_depth=pred_depth_map.cpu(),
        #     #               gt_depth=data['gt_depth'],
        #     #               confidence_map_rgb=conf1.cpu(),
        #     #               confidence_map_depth=conf2.cpu(),
        #     #               depth_map_rgb=depth1.cpu(),
        #     #               depth_map_depth=depth2.cpu())
        #     # val_dataloader.dataset.show(sample, i, args.sample_dir, save_array=True)
        #
        # generator.train()
        exit(0)

    # build loss
    from lib.losses.rdf_gan_loss import GANLoss, L1_loss
    l1_loss_coef = args.l1_loss_coef
    generator_loss_coef = 1.0

    l1_loss = L1_loss
    gan_loss = GANLoss(gan_mode=args.gan_loss_type).to(device)

    _iters = start_epoch * num_iters_per_epoch

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = args.warm_up_steps * len(train_dataloader)

    for epoch in range(start_epoch, args.max_epoch):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)

        step_losses = dict()

        for i, data in enumerate(train_dataloader):
            generator.train()
            discriminator_rgb.train()
            discriminator_depth.train()

            if epoch < args.warm_up_steps and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer_generator.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

                for param_group in optimizer_disc_rgb.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

                for param_group in optimizer_disc_depth.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            # fetch a batch of data
            rgb = data['rgb'].to(device)
            depth = data['raw_depth'].to(device)
            depth_undistorted = data['gt_depth'].to(device)
            if 'depth_masks' in data:
                # whether ignore the invalid pixel or not
                mask = data['depth_masks'].to(device)
            else:
                mask = torch.ones_like(depth).to(device)

            l1_loss_weight = mask / (mask.sum() + EPSION)

            loss_stats = dict()


            #---------------------------------- train discriminator ---------------------------------- #
            requires_grad(generator, False)
            requires_grad(discriminator_rgb, True)
            requires_grad(discriminator_depth, True)
            depth_map_1, confidence_map_1, depth_map_2, confidence_map_2, final_depth_map = generator(rgb, depth)


            real_img, fake_img = depth_undistorted, depth_map_1
            # 1. train descriminator rgb branch
            discriminator_rgb.zero_grad()
            # real
            pred_real = discriminator_rgb(real_img)
            weight = torch.ones_like(pred_real).to(device)
            disc_rgb_real_loss = gan_loss(pred_real, True, weight=weight / (weight.sum() + EPSION))
            disc_rgb_real_loss.backward()

            # fake
            # use .detach() to cut off the gradient propagation to generator,[unnecessary, already setting gardient to False]
            pred_fake = discriminator_rgb(fake_img)
            weight = torch.ones_like(pred_fake).to(device)
            disc_rgb_fake_loss = gan_loss(pred_fake, False, weight=weight / (weight.sum() + EPSION))
            disc_rgb_fake_loss.backward()

            if args.gan_loss_type == 'wgangp':
                # I didn't implement gardient-penalty in GANLoss helper class
                # DistributedDataParallel not work with torch.autograd.grad()
                eps = torch.rand(real_img.shape[0], 1, 1, 1).to(device)
                x_hat = eps * real_img.data + (1 - eps) * fake_img.data
                x_hat.requires_grad = True
                hat_predict = discriminator_rgb(x_hat)
                grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
                grad_penalty = (
                        (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
                ).mean()
                grad_penalty = 10 * grad_penalty  # the lambda coefficient equals to 10 in wgan-gp
                grad_penalty.backward()
                loss_stats.update({'grad_penalty_rgb': grad_penalty})

            loss_stats.update(dict(disc_rgb_fake_loss=disc_rgb_fake_loss,
                                   disc_rgb_real_loss=disc_rgb_real_loss,
                                   disc_rgb_loss=disc_rgb_fake_loss + disc_rgb_real_loss))

            optimizer_disc_rgb.step()

            # train discriminator_depth
            real_img, fake_img = depth_undistorted, depth_map_2
            discriminator_depth.zero_grad()

            # real
            pred_real = discriminator_depth(real_img)
            weight = torch.ones_like(pred_real).to(device)
            disc_depth_real_loss = gan_loss(pred_real, True, weight=weight / (weight.sum() + EPSION))
            disc_depth_real_loss.backward()

            pred_fake = discriminator_depth(depth_map_2)
            weight = torch.ones_like(pred_fake).to(device)
            disc_depth_fake_loss = gan_loss(pred_fake, False, weight=weight / (weight.sum() + EPSION))
            disc_depth_fake_loss.backward()
            if args.gan_loss_type == 'wgangp':
                eps = torch.rand(real_img.shape[0], 1, 1, 1).to(device)
                x_hat = eps * real_img.data + (1 - eps) * fake_img.data
                x_hat.requires_grad = True
                hat_predict = discriminator_depth(x_hat)
                grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
                grad_penalty = (
                        (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
                ).mean()
                grad_penalty = 10 * grad_penalty  # the lambda coefficient equals to 10 in wgan-gp
                grad_penalty.backward()
                loss_stats.update({'grad_penalty_depth': grad_penalty})
            loss_stats.update(dict(disc_depth_fake_loss=disc_depth_fake_loss,
                                   disc_depth_real_loss=disc_depth_real_loss,
                                   disc_depth_loss=disc_depth_fake_loss + disc_depth_real_loss))

            optimizer_disc_depth.step()

            if args.gan_loss_type == 'wgan':
                for param in discriminator_rgb.parameters():
                    param.data.clamp_(-args.wgan_clip_value, args.wgan_clip_value)


            #---------------------------------- train generator ---------------------------------- #
            if (_iters + 1) % args.n_critic == 0:
                generator.zero_grad()
                requires_grad(generator, True)
                requires_grad(discriminator_rgb, False)
                requires_grad(discriminator_depth, False)

                depth_map_1, confidence_map_1, depth_map_2, confidence_map_2, final_depth_map = generator(rgb, depth)

                pred_fake_1 = discriminator_rgb(depth_map_1)
                pred_fake_2 = discriminator_depth(depth_map_2)
                weight = torch.ones_like(pred_fake_1).to(device)
                gen_rgb_fake_loss = gan_loss(pred_fake_1, True, weight=weight / (weight.sum() + EPSION))
                weight = torch.ones_like(pred_fake_2).to(device)
                gen_depth_fake_loss = gan_loss(pred_fake_2, True, weight=weight / (weight.sum() + EPSION))
                gen_gan_loss = generator_loss_coef * (gen_rgb_fake_loss + gen_depth_fake_loss)

                gen_l1_loss = l1_loss_coef * l1_loss(final_depth_map, depth_undistorted, weight=l1_loss_weight)

                gen_total_loss = gen_gan_loss + gen_l1_loss
                loss_stats.update(dict(gen_rgb_fake_loss=gen_rgb_fake_loss,
                                       gen_depth_fake_loss=gen_depth_fake_loss,
                                       gen_gan_loss=gen_gan_loss,
                                       gen_l1_loss=gen_l1_loss,
                                       gen_total_loss=gen_total_loss))

                gen_total_loss.backward()
                optimizer_generator.step()

                requires_grad(generator, False)
                requires_grad(discriminator_rgb, True)
                requires_grad(discriminator_depth, True)

            reduced_loss = reduce_loss(loss_stats)
            for k in reduced_loss:
                if k not in step_losses:
                    step_losses[k] = MovingAverage(reduced_loss[k],
                                                   window_size=args.log_interval if "gen" not in k else args.log_interval // args.n_critic)
                else:
                    step_losses[k].push(reduced_loss[k])


            if (i + 1) % args.log_interval == 0:
                log_msg = "Train - Epoch [{}][{}/{}] ".format(epoch + 1,
                                                              i + 1,
                                                              num_iters_per_epoch)

                for name in step_losses:
                    val = step_losses[name].avg()
                    log_msg += "{}: {:.4f}| ".format(name, val)
                    logger.scalar_summary(name, val, _iters + 1)

                lr_generator = [group['lr'] for group in optimizer_generator.param_groups]
                lr_disc_rgb = [group['lr'] for group in optimizer_disc_rgb.param_groups]
                lr_disc_depth = [group['lr'] for group in optimizer_disc_depth.param_groups]
                logger.scalar_summary(f'lr_generator', lr_generator[0], _iters + 1)
                logger.scalar_summary(f'lr_disc_rgb', lr_disc_rgb[0], _iters + 1)
                logger.scalar_summary(f'lr_disc_depth', lr_disc_depth[0], _iters + 1)

                logger.log(log_msg)

            if (_iters + 1) % args.sample_interval == 0:
                if args.local_rank == 0:
                    generator.eval()
                    with torch.no_grad():
                        depth1, conf1, depth2, conf2, pred_depth_map = generator(fixed_samples['rgb'].to(device),
                                                                                 fixed_samples['raw_depth'].to(device))

                    sample = dict(rgb=fixed_samples['rgb'],
                                  raw_depth=fixed_samples['raw_depth'],
                                  pred_depth=pred_depth_map.cpu(),
                                  gt_depth=fixed_samples['gt_depth'])
                    val_dataloader.dataset.show(sample, _iters + 1, args.sample_dir)
                    generator.train()

            _iters += 1

        del step_losses

        if not (epoch < args.warm_up_steps and args.warm_up):
            lr_scheduler_generator.step()
            lr_scheduler_disc_rgb.step()
            lr_scheduler_disc_depth.step()

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.max_epoch:
            # Epoch based
            if args.local_rank == 0:
                save_checkpoint(module=dict(generator=generator,
                                            disc_rgb=discriminator_rgb,
                                            disc_depth=discriminator_depth),
                                filename=os.path.join(args.work_dir, f'epoch_{epoch + 1}.pth'),
                                optimizer=dict(generator=optimizer_generator,
                                               disc_rgb=optimizer_disc_rgb,
                                               disc_depth=optimizer_disc_depth),
                                lr_scheduler=dict(generator=lr_scheduler_generator,
                                                  disc_rgb=lr_scheduler_disc_rgb,
                                                  disc_depth=lr_scheduler_disc_depth),
                                meta=dict(epoch=epoch + 1))


        if (epoch + 1) >= args.start_eval_epoch and (epoch + 1) % args.val_interval == 0:
            evaluator.evaluate(generator, logger)


if __name__ == '__main__':
    main()
