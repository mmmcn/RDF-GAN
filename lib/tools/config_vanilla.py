"""
    Author: wmy

    All of the parameters are defined here.
"""
import os
import argparse


parser = argparse.ArgumentParser(description='RDF-GAN, vanilla version')



parser.add_argument('--height', type=int, default=480)
parser.add_argument('--width', type=int, default=640)

# model
## guidance model
parser.add_argument('--pretrained_on_imagenet', default=False, action='store_true')
parser.add_argument('--guidance_pretrained_dir', type=str, default='./pretrained_model/resnet_on_imagenet')
parser.add_argument('--guidance_encoder', type=str, default='resnet50')
parser.add_argument('--guidance_encoder_block', type=str, default='Bottleneck'),
parser.add_argument('--guidance_channels_decoder', type=int, default=[512, 256, 128], nargs='+')
parser.add_argument('--guidance_nr_decoder_blocks', type=int, default=[3, 3, 3], nargs='+')
parser.add_argument('--guidance_encoder_decoder_fusion', type=str, default='add')
parser.add_argument('--guidance_context_module', type=str, default='ppm')
parser.add_argument('--guidance_weighting_in_encoder', type=str, default='SE-add')
parser.add_argument('--guidance_upsampling', type=str, default='learned-3x3-zeropad')

## other model
parser.add_argument('--reduction_glo_guid_module', action='store_true')
parser.add_argument('--glo_guid_channels_out_0', type=int, default=1)
parser.add_argument('--glo_guid_channels_out_1', type=int, default=1)
parser.add_argument('--encoder_rgb', type=str, default='resnet18')
parser.add_argument('--encoder_depth', type=str, default='resnet18')
parser.add_argument('--encoder_block', type=str, default='BasicBlock')
parser.add_argument('--rgb_channels_decoder', type=int, default=[512, 256, 128, 64, 64], nargs='+')
parser.add_argument('--depth_channels_decoder', type=int, default=[512, 256, 128, 64, 64], nargs='+')
parser.add_argument('--nr_decoder_blocks', type=int, default=[3, 3, 3, 0, 0], nargs='+')
parser.add_argument('--fuse_depth_in_rgb_encoder',
                    type=str,
                    default='None')
parser.add_argument('--fuse_depth_in_rgb_decoder',
                    type=str,
                    default='AdaIN',
                    choices=['AdaIN', 'None'])
parser.add_argument('--encoder_decoder_fusion', type=str,
                    default='add',
                    choices=['add', 'None'],
                    help='How to fuse encoder feature maps into the'
                         'decoder. If None no encoder feature maps are'
                         'fused into the decoder')           # control rgb branch, for depth branch, no fuse is used
parser.add_argument('--activation',
                    type=str,
                    default='LeakyReLU',
                    choices=['LeakyReLU', 'ReLU'],
                    help='Which activation function to use in the model')
parser.add_argument('--negative_slope',
                    type=float,
                    default=0.02,
                    help='The negative_slope hyper parameter for LeakyRelU activation function')
parser.add_argument('--norm_layer_type',
                    type=str,
                    default='IN2d',
                    choices=['IN2d', 'BN2d'],
                    help='Which normalization layer to use in the model')
parser.add_argument('--upsampling_mode',
                    type=str,
                    default='bilinear',
                    choices=['nearest', 'bilinear', 'learned-3x3', 'learned-3x3-zeropad'],
                    help='How to usample in the decoder. '
                         'Bilinear upsampling can cause problems'
                         'with conversion to TensorRT. learned-3x3 '
                         'mimics a bilinear interpolation with nearest '
                         'neighbor interpolation and a 3x3 conv '
                         'afterwards')
parser.add_argument('--adain_weighting', action='store_true')

parser.add_argument('--disc_norm_type', type=str, default='bn')
parser.add_argument('--disc_act_type', type=str, default='relu')

# dataset
parser.add_argument('--dataset', type=str, default='sunrgbd',
                    choices=['sunrgbd', 'nyudepthv2_s2d', 'nyudepthv2_r2r',
                             'nyuv21400_s2d'],
                    help='dataset name')
parser.add_argument('--data_root', type=str,
                    default='data/sunrgbd',
                    help='path to dataset')
parser.add_argument('--num_classes', type=int, default=37)
parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU during training')
parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')


# training
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--n_critic', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='optimizer')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 coefficient used for computing running averages of gradient and its square in adam')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 coefficient used for computing running averages of gradient and its square in adam')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Optimization L2 weight decay [default: 0.0001]')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='Initial learning rate')
parser.add_argument('--lr_scheduler', type=str, default='cosine',
                    choices=["step", "cosine", "onecycle"], help="learning rate scheduler")
parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                    help='for step scheduler. where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                    help='for step scheduler. decay rate for learning rate')
parser.add_argument('--t_max', type=int, default=250,
                    help='for cosine annealing scheduler, maximum number of iterations')
parser.add_argument('--div_factor', type=float, default=25.0,
                    help='div_factor for onecycle scheduler')
parser.add_argument('--pct_start', type=float, default=0.1,
                    help='pct_start for onecycle scheduler')
parser.add_argument('--gan_loss_type', type=str, default='wgangp',
                    choices=['wgan', 'wgangp', 'lsgan', 'vanilla'],
                    help='class of gan loss')
parser.add_argument('--wgan_clip_value', type=float, default=0.01)
parser.add_argument('--use_pretrained_encoder_decoder', action='store_true')
parser.add_argument('--load_encoder_decoder_path',
                    type=str,
                    default='./work_dir/train_segmentator/esa_one_modal_rgb_37/best.pth')
parser.add_argument('--freeze_encoder_decoder', action='store')

parser.add_argument('--inference', action='store_true')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=False,
                    help='whether use lr warm up')
parser.add_argument('--warm_up_steps',
                    type=int,
                    default=1,
                    help='number of epoch to do lr warm up')

# loss
parser.add_argument('--l1_loss_coef', type=float, default=100.0)

# io
parser.add_argument('--work_dir', default='./', help='the dir to save logs and models')
parser.add_argument('--resume_from', default=None, help='ckpt file path to resume from')
parser.add_argument('--load_from', default=None, help='ckpt file path to load from')
parser.add_argument('--log_interval', type=int, default=30, help='log msg frequency')
parser.add_argument('--save_interval', type=int, default=2, help='ckpt saving frequency')
parser.add_argument('--sample_interval', type=int, default=100, help='val frequency, visiualization')
parser.add_argument('--val_interval', type=int, default=2)
parser.add_argument('--start_eval_epoch', type=int, default=1)
parser.add_argument('--sample_dir', type=str, default='./')

# seed, rank and others
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpus', type=str, default="0", help="gpus to use")
parser.add_argument("--local_rank", type=int, default=0, help='local rank for distributed training')

parser.add_argument('--separate_global_guidance_module', action='store_true')
parser.add_argument('--init_disc', action='store_true')

parser.add_argument('--cal_fps', default=False, action='store_true')

args = parser.parse_args()
args.num_gpus = len(args.gpus.split(','))

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)