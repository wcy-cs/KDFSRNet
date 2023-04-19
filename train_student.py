import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
from model import net
from data import dataset_parsing
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util


epochs = args.epochs
lr = args.lr
student = net.Student(args)
teacher = net.Teacherp(args)

pretrained_dict = torch.load(args.teacher_path, map_location='cuda:0')

teacher.load_state_dict(pretrained_dict)
for par in teacher.parameters():
    par.requires_grad = False

teacher = util.prepare(teacher)
student = util.prepare(student)

print(util.get_parameter_number(student))
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
traindata = dataset_parsing.Data(root=os.path.join(args.dir_data, args.data_train), args=args, train=True)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=16)
valdata = dataset_parsing.Data(root=os.path.join(args.dir_data, args.data_val), args=args, train=False)
valset = DataLoader(valdata, batch_size=1, shuffle=False, num_workers=1)
criterion1 = nn.L1Loss()
optimizer = optim.Adam(params=student.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)


adaptive_weight1_inter = 10
for i in range(epochs):
    student.train()
    train_loss = 0
    bum = len(trainset)
    for batch, (lr, hr, parsing, _) in enumerate(trainset):
        lr, hr, parsing = util.prepare(lr), util.prepare(hr), util.prepare(parsing)
        inter_teacher, srt = teacher(lr, parsing)
        inter_student, sr = student(lr)
        l1_loss = criterion1(sr, hr)
        dist_loss = 0
        for k in range(9):
            dist_loss = dist_loss + criterion1(inter_student[k], inter_teacher[k])
        dist_loss = dist_loss / 9
        soft_loss_num = criterion1(srt, sr)

        if i > args.adaptive_weight1_inter:
            loss = args.l1_weight * l1_loss
        else:
            loss = args.l1_weight * l1_loss + args.dist_weight * (adaptive_weight1_inter - i) / (
                        adaptive_weight1_inter) * dist_loss  + args.soft_weight * soft_loss_num
        train_loss = train_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch：{} loss: {:.3f}".format(i + 1, train_loss / (len(trainset)) * 255))
    writer.add_scalar('train_loss', train_loss / (len(trainset)) * 255, i + 1)
    os.makedirs(os.path.join(args.save_path, args.writer_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'model'), exist_ok=True)
    torch.save(student.state_dict(),
                   os.path.join(args.save_path, args.writer_name, 'model', 'epoch{}.pth'.format(i + 1)))
    student.eval()
    val_psnr = 0
    val_ssim = 0
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)
    for batch, (lr, hr, parsing, filename) in enumerate(valset):
        lr, hr, parsing = util.prepare(lr), util.prepare(hr), util.prepare(parsing)
        _, sr = student(lr)
        psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())
        val_psnr = val_psnr + psnr_c
        val_ssim = val_ssim + ssim_c
    print("Epoch：{} val  psnr: {:.3f}".format(i + 1, val_psnr / (len(valset))))
    writer.add_scalar("val_psnr_DIC", val_psnr / len(valset), i + 1)
    writer.add_scalar("val_ssim_DIC", val_ssim / len(valset), i + 1)








