from option import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim

import torch.nn as nn
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util
import torchvision
from model import network
import numpy as n
epochs = args.epochs
student = network.Student(args)
student = util.prepare(student)


print(util.get_parameter_number(student))


testdata = dataset.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)

pretrained_dict = torch.load(args.load,map_location='cuda:0')

        
                
student.load_state_dict(pretrained_dict)
student = util.prepare(student)

student.eval()
val_psnr = 0
with torch.no_grad():
    save_name= "result-test"
    if "helen" in args.dir_data:
        save_name = 'result-test-helen'
    elif "FFHQ" in args.dir_data:
        save_name = 'result-test-FFHQ'
    elif "CelebA" in args.dir_data:
        save_name = 'result-test-CelebA'
    else:
        save_name = "result-test"

    os.makedirs(os.path.join(args.save_path, args.writer_name, save_name), exist_ok=True)
    student.eval()
    timer_test = util.timer()
    for batch, (lr, hr, filename) in enumerate(testset):

        lr, hr = util.prepare(lr), util.prepare(hr)
        _, sr = student(lr)

        psnr1, _ = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu(), crop_border=8)
        val_psnr = val_psnr + psnr1

        torchvision.utils.save_image(sr[0],
                                         os.path.join(args.save_path, args.writer_name, save_name,
                                                      '{}'.format(str(filename[0])[:-4] + ".png")))
    print("Test psnr: {:.3f}".format(val_psnr / (len(testset))))
    print('Forward: {:.2f}s\n'.format(timer_test.toc()))
  












