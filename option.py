import argparse
# import os

parser = argparse.ArgumentParser(description='FaceSR')

parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--cuda_name', type=str, default='1')
parser.add_argument('--gpu_ids', type=int, default=1)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--dir_data', type=str, default='/userhome/data/CelebA',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='train',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',
                    help='test dataset name')
parser.add_argument('--data_val', type=str, default='val',
                    help='val dataset name')
parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--augment', action='store_true',
                    help='use data augmentation')
parser.add_argument('--large_parsing', action="store_true",
                    help='Coor')
# Model specifications
parser.add_argument('--model', default='AWNET',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.2,
                    help='residual scaling')
parser.add_argument('--skin', action="store_true",
                    help='residual scaling')
parser.add_argument('--PCSR', action="store_true",
                    help='Coor')
parser.add_argument('--PCSR1', action="store_true",
                    help='Coor')
# TwoBranch Model specifications
parser.add_argument('--add_feature', action="store_true",
                    help='add feature from heatmap to sr branch')

parser.add_argument('--concat_input', action="store_true",
                    help='concat parsing map in fish input')

parser.add_argument('--concat_head1', action="store_true",
                    help='concat parsing map in fish head')

parser.add_argument('--concat_body1', action="store_true",
                    help='concat parsing map in fish body')

parser.add_argument('--concat_tail1', action="store_true",
                    help='concat parsing map in fish tail')

parser.add_argument('--concat_head2', action="store_true",
                    help='concat parsing map in fish head')

parser.add_argument('--concat_body2', action="store_true",
                    help='concat parsing map in fish body')

parser.add_argument('--concat_tail2', action="store_true",
                    help='concat parsing map in fish tail')

parser.add_argument('--concat_head3', action="store_true",
                    help='concat parsing map in fish head')

parser.add_argument('--concat_body3', action="store_true",
                    help='concat parsing map in fish body')

parser.add_argument('--concat_tail3', action="store_true",
                    help='concat parsing map in fish tail')

parser.add_argument('--concat_head4', action="store_true",
                    help='concat parsing map in fish head')

parser.add_argument('--concat_body4', action="store_true",
                    help='concat parsing map in fish body')

parser.add_argument('--concat_tail4', action="store_true",
                    help='concat parsing map in fish tail')
parser.add_argument('--CSCR', action="store_true", help='CSCR')
parser.add_argument('--CSCR1', action="store_true",
                    help='CSCR1')
parser.add_argument('--CA', action="store_true", help='CSCR')
parser.add_argument('--SA', action="store_true",
                    help='CSCR1')
parser.add_argument('--multi', action="store_true",
                    help='use multi-scale conv')
parser.add_argument('--spatial', action="store_true",
                    help='use multi-scale conv')
parser.add_argument('--refine', action="store_true",
                    help='use multi-scale conv')
parser.add_argument('--refine2', action="store_true",
                    help='use multi-scale conv')
parser.add_argument('--refine1', action="store_true",
                    help='use multi-scale conv')
parser.add_argument('--in_dim', type=int, default=18,
                    help='use residual  and conv relu')
parser.add_argument('--out_dim', type=int, default=50,
                    help='use residual  and conv relu')
parser.add_argument('--use_attri', action='store_true',
                    help='use attribute information as a branch')
parser.add_argument('--con_attri', action='store_true',
                    help='concat attribute before the network')

parser.add_argument('--large', action="store_true",
                    help='the input is as large as output or not')

parser.add_argument('--vgg_pretrain', action="store_true",
                    help='the input is as large as output or not')

parser.add_argument('--bicubic', action="store_true",
                    help='magnify feature by bicubic')
# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',#default=True,
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',# default=True,
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--n_steps', type=int, default=30,
                    help='学习率衰减倍数')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='l1',
                    help='loss function configuration')



# SPL parameter
parser.add_argument('--splalpha', default=2,
                    help='sigma init value')
parser.add_argument('--splbeta', default=0.2,
                    help='sigma init value')
parser.add_argument('--splval', default=2,
                    help='sigma init value')

# Log specifications
parser.add_argument('--root', type=str, default='/userhome/experiment')
parser.add_argument('--save', type=str, default='mynet',
                    help='file name to save')
parser.add_argument('--save_test', type=str, default='awnet',
                    help='file name to save test result')
parser.add_argument('--save_path', type=str, default='/userhome/experiment',
                    help='file path to save model')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true', default=True,
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=True,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument("--writer_name", type=str, default="mynet",
                    help="the name of the writer")

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))


if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

