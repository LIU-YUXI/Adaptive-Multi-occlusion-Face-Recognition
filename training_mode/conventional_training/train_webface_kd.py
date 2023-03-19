"""
@author: Yuxi Liu
@date: 20230218
@contact: liuyuxi_tongji@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import os.path as osp
sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset,ImageDataset_KD
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from backbone.iresnet import iresnet100
logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat, label)
        return pred

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, backbone_teacher, backbone_student, header, model_optimizer, header_optimizer,
                        criterion,criterion2, cur_epoch, loss,loss1,loss2, conf):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (mask_images,images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        mask_images = mask_images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        with torch.no_grad():
            features_teacher = F.normalize(backbone_teacher(images))
        features_student = F.normalize(backbone_student(mask_images))

        thetas = header(features_student, labels)
        loss_v1 = criterion(thetas, labels)
        loss_v2 = conf.w*criterion2(features_student, features_teacher)
        loss_v = loss_v1 + loss_v2
        loss_v.backward()#compute

        clip_grad_norm_(backbone_student.parameters(), max_norm=5, norm_type=2)

        model_optimizer.step()
        header_optimizer.step()#update

        model_optimizer.zero_grad()
        header_optimizer.zero_grad()

        loss.update(loss_v.item(), 1)
        loss1.update(loss_v1.item(), 1)
        loss2.update(loss_v2.item(), 1)        

        if batch_idx % conf.print_freq == 0:
            loss_avg = loss.avg
            lr = get_lr(model_optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss.reset()
        if (batch_idx + 1) % conf.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': backbone_student.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            saved_name_header = 'Epoch_%d_batch_%d_header.pt' % (cur_epoch, batch_idx)
            state_header = {
                'state_dict': header.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state_header, os.path.join(conf.out_dir, saved_name_header))
            logger.info('Save checkpoint %s to disk.' % saved_name)
    saved_name = 'Epoch_%d.pt' % cur_epoch
    state = {'state_dict': backbone_student.module.state_dict(), 
             'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(conf.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

def train(conf):
    """Total training procedure.
    """
    transform = transforms.Compose(            
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])

    # casia_path = conf.data_root# osp.join(root, 'casia_webface')
    # dataset = dsets.ImageFolder(root=casia_path, transform=transform)

    # print('classes',dataset.classes)  
    # print('indexs',dataset.class_to_idx) 
    # print('images',dataset.imgs[0]) 
    # data_loader = torch.utils.data.DataLoader(dataset, 
    #                                    batch_size=conf.batch_size, 
    #                                    shuffle=True, num_workers=4, pin_memory = True)
    data_loader = DataLoader(ImageDataset_KD(conf.data_root, conf.train_file,transform=transform), 
                               conf.batch_size, True, num_workers = 4)
    conf.device = torch.device('cuda:0')
    backbone_teacher = iresnet100(num_features=conf.embedding_size)
    try:
        backbone_teacher_pth = os.path.join(conf.teacher_pth, str(conf.teacher_global_step) + "backbone.pth")
        backbone_teacher.load_state_dict(torch.load(backbone_teacher_pth))
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logger.info("load teacher backbone init, failed!")
    # load student model
    backbone_student = iresnet100(num_featuresz=conf.embedding_size)

    if conf.pretrained_student:
        try:
            #backbone_student_pth = os.path.join(conf.student_pth, str(conf.student_global_step) + "backbone.pth")
            backbone_student.load_state_dict(torch.load(conf.pretrain_model))
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logger.info("load student backbone init, failed!")
    if conf.resume:
        try:
            backbone_student_pth = os.path.join(conf.out_dir,conf.pretrain_model )
            backbone_student.load_state_dict(torch.load(backbone_student_pth))
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logger.info("load student backbone resume init, failed!")
    backbone_teacher.eval()
    # backbone_student.train()
    # backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)    
    # header = HeadFactory(conf.head_type, conf.head_conf_file).get_head()
    header = ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m)
    if conf.resume:
        try:
            backbone_header_pth = os.path.join(conf.out_dir,conf.pretrain_header )
            header.load_state_dict(torch.load(backbone_header_pth))
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logger.info("load header backbone resume init, failed!")
    # model = FaceModel(backbone_factory, head_factory)
    ori_epoch = 0
    if conf.resume:
        ori_epoch = torch.load(args.pretrain_model)['epoch'] + 1
    model = torch.nn.DataParallel(backbone_student).cuda()
    parameters = [p for p in backbone_student.parameters() if p.requires_grad]
    model_optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule_model = optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones = conf.milestones, gamma = 0.1)
    # loss_meter = AverageMeter()
    loss = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    # This function computes the average loss over an epoch, that is, the average of the loss over each sample.
    model.train()
    header=torch.nn.DataParallel(header).cuda()
    header_optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule_header = optim.lr_scheduler.MultiStepLR(
        header_optimizer, milestones = conf.milestones, gamma = 0.1)
    header.train()
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    criterion2 = torch.nn.MSELoss().cuda(conf.device)
    for epoch in range(ori_epoch, conf.epoches):
        train_one_epoch(data_loader, backbone_teacher, model,header, model_optimizer, header_optimizer,
                        criterion,criterion2, epoch, loss,loss1,loss2, conf)
        lr_schedule_header.step()  
        lr_schedule_model.step()            

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str, 
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str, 
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, default='../../output/',
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '2,5,7', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--embedding_size', type = int, default = 512, 
                      help='embedding size of backbones.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--w', type = float, default = 100, 
                      help = 'weight for MSE loss.')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--pretrain_header', type = str, default = 'mv_epoch_8_header.pt', 
                      help = 'The path of pretrained header')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')
    conf.add_argument("--teacher_pth", type = str, default = '/root/lyx/maskinv/output/emore_random_resnet',
                      help = "the path of teacher backbone.")
    conf.add_argument("--teacher_global_step", type = int , default = 295672,
                      help = "the step of teacher backbone.")       
    conf.add_argument("--teacher_network", type = str , default = 'resnet100',
                      help = "the name of teacher backbone.")                                    
    conf.add_argument("--student_pth", type = str, default = '/root/lyx/maskinv/output/emore_random_resnet_student',
                      help = "the path of teacher backbone.")
    conf.add_argument("--student_global_step", type = int , default = 0,
                      help = "the step of teacher backbone.")       
    conf.add_argument("--student_network", type = str , default = 'resnet100',
                      help = "the name of teacher backbone.")                    
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
