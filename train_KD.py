# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import torchnet as tnt


from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import dataset
from torchvision.models import resnext50_32x4d

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dataset_stanford_KD import get_loader_KD
from utils.dist_util import get_world_size
import torchmetrics

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt


def visualize_model(args, model, testloader, class_names, num_images=4):
    model.eval()
    images_so_far = 0
    # fig = plt.figure()

    with torch.no_grad():
        for i, (x1, x2, labels) in enumerate(testloader):
            
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            labels = labels.to(args.device)

            outputs = model(x1, x2)
            _, preds = torch.max(outputs, 1)

            fig, axs = plt.subplots(2, 2)

            # for j in range(inputs.size()[0]): 
            # print(axs)
            for j, ax in enumerate(axs.flat):
                ax.axis('off')
                # print(j,ax)
                if j > x1.size()[0]:
                    return
                images_so_far += 1
                ax.set_title('p:{}\ng:{}'.format(class_names[preds[j]], class_names[labels[j]]), fontsize='small', loc='left')
                image = np.moveaxis(np.asarray(x1.data[j].squeeze().cpu().detach()), 0, -1)
                # np.asarray
                im = ax.imshow(image)
            # plt.show()
            plt.savefig("/content/myimage{}_{}.png".format(i,j))
        return


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return (self.sigmoid(out).expand_as(x) + 1) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w = x.size()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return (self.sigmoid(out).expand_as(x) + 1) * x

class EnsembleModel(nn.Module):
  def __init__(self):
    super(EnsembleModel, self).__init__()
    # self.models = []
    self.k = 2
    # first model
    model_1 = resnext50_32x4d(pretrained=True, progress=True)
    model_1.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
    lay4 = list(model_1.layer4)
    channel0_1 = ChannelAttention(2048,16)
    spatial0_1 = SpatialAttention()
    # bn0_1 = nn.BatchNorm2d(2048)
    channel1_1 = ChannelAttention(2048,16)
    spatial1_1 = SpatialAttention()
    # bn1_1 = nn.BatchNorm2d(2048)
    channel2_1 = ChannelAttention(2048,16)
    spatial2_1 = SpatialAttention()
    # bn2_1 = nn.BatchNorm2d(2048)
    # layer4_new = [lay4[0], channel0_1, spatial0_1, bn0_1, lay4[1], channel1_1, spatial1_1, bn1_1, lay4[2], channel2_1, spatial2_1, bn2_1]
    layer4_new = [lay4[0], channel0_1, spatial0_1, lay4[1], channel1_1, spatial1_1, lay4[2], channel2_1, spatial2_1]
    model_1.layer4 = nn.Sequential(*layer4_new)
    # ckpt_1 = torch.load('/content/drive/MyDrive/KD_ResNext_ViT_weights/best_acc_step_800_acc_0.8533984092552422_checkpoint.pth')
    # model_1.load_state_dict(ckpt_1['model_state_dict'], strict=True)
    # logger.info('acc of the first model is {}.'.format(ckpt_1['best_accuracy']))
    self.model_1 = model_1

    # second model
    model_2 = resnext50_32x4d(pretrained=True, progress=True)
    model_2.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
    # ckpt_2 = torch.load('/content/drive/MyDrive/ResNeXT/Copy of ckpt30.pth')
    # model_2.load_state_dict(ckpt_2['model'], strict= True)
    # logger.info('acc of the second model is {}.'.format(ckpt_2['acc']))
    lay4 = list(model_2.layer4)
    channel0 = ChannelAttention(2048,16)
    spatial0 = SpatialAttention()
    # bn0_2 = nn.BatchNorm2d(2048)
    channel1 = ChannelAttention(2048,16)
    spatial1 = SpatialAttention()
    # bn1_2 = nn.BatchNorm2d(2048)
    channel2 = ChannelAttention(2048,16)
    spatial2 = SpatialAttention()
    # bn2_2 = nn.BatchNorm2d(2048)
    # layer4_new = [lay4[0], channel0, spatial0, bn0_2, lay4[1], channel1, spatial1, bn1_2, lay4[2], channel2, spatial2, bn2_2]
    layer4_new = [lay4[0], channel0, spatial0, lay4[1], channel1, spatial1, lay4[2], channel2, spatial2]
    model_2.layer4 = nn.Sequential(*layer4_new)
    # ckpt_2 = torch.load('/content/drive/MyDrive/KD_First_weights/KD_ResNext_ViT_weights/ckpt1_acc_0.8617136478424072.pth')
    # model_2.load_state_dict(ckpt_2['model'], strict=False)
    # logger.info('acc of the second model is {}.'.format(ckpt_2['acc']))
    self.model_2 = model_2
    self.relu_last = nn.ReLU()
    self.classifier = nn.Linear(80, 40, True)

  def forward(self, x):
    x1 = self.model_1(x)
    x2 = self.model_2(x)
    y = torch.cat((x1, x2), dim=1)
    # print(y.shape)
    y = self.classifier(self.relu_last(y))
    # y = torch.stack((x1, x2), dim = 1).mean(dim = 1)
    return y


# class EnsembleModel_resnext_vit(nn.Module):
#   def __init__(self, args, config, img_size=224, zero_head=True, num_classes=40):
#     super().__init__()
#     self.k = 2
#     # first model
#     model_1 = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
#     checkpoint_file = args.input_dir
#     checkpoint = torch.load(checkpoint_file)
#     if 'model_state_dict' in checkpoint.keys():
#         model_1.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model_1.load_state_dict(checkpoint)
#     self.model_1 = model_1

#     # second model
#     model_2 = resnext50_32x4d(pretrained=True, progress=True)
#     model_2.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
#     # ckpt_2 = torch.load('/content/drive/MyDrive/ResNeXT/Copy of ckpt30.pth')
#     # model_2.load_state_dict(ckpt_2['model'], strict= True)
#     # logger.info('acc of the second model is {}.'.format(ckpt_2['acc']))
#     lay4 = list(model_2.layer4)
#     channel0 = ChannelAttention(2048,16)
#     spatial0 = SpatialAttention()
#     # bn0_2 = nn.BatchNorm2d(2048)
#     channel1 = ChannelAttention(2048,16)
#     spatial1 = SpatialAttention()
#     # bn1_2 = nn.BatchNorm2d(2048)
#     channel2 = ChannelAttention(2048,16)
#     spatial2 = SpatialAttention()
#     # bn2_2 = nn.BatchNorm2d(2048)
#     # layer4_new = [lay4[0], channel0, spatial0, bn0_2, lay4[1], channel1, spatial1, bn1_2, lay4[2], channel2, spatial2, bn2_2]
#     layer4_new = [lay4[0], channel0, spatial0, lay4[1], channel1, spatial1, lay4[2], channel2, spatial2]
#     model_2.layer4 = nn.Sequential(*layer4_new)
#     # ckpt_2 = torch.load('/content/drive/MyDrive/KD_First_weights/KD_ResNext_ViT_weights/ckpt1_acc_0.8617136478424072.pth')
#     ckpt_2 = torch.load('/content/drive/MyDrive/ResNext_cbam/TrainedModels/ResNext_cbam/Adam/_lr_1e-06/_wd_0.0/ckpt164_acc_0.8781633973121643.pth')
#     model_2.load_state_dict(ckpt_2['model'], strict=False)
#     logger.info('acc of the second model is {}.'.format(ckpt_2['acc']))
#     self.model_2 = model_2
#     self.relu_last = nn.ReLU()
#     self.classifier = nn.Linear(80, 40, True)

#   def forward(self, x1, x2):
#     y1, _ = self.model_1(x1)
#     y2 = self.model_2(x2)
#     y = torch.cat((y1, y2), dim=1)
#     # print(y.shape)
#     y = self.classifier(self.relu_last(y))
#     # y = torch.stack((y1, y2), dim = 1).mean(dim = 1)
#     return y

# class EnsembleModel_resnext_vit(nn.Module):
#     def __init__(self, args, config, img_size=224, zero_head=True, num_classes=40):
#         super().__init__()
#         self.k = 2
#         # first model
#         model_1 = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
#         checkpoint_file = args.input_dir
#         checkpoint = torch.load(checkpoint_file)
#         if 'model_state_dict' in checkpoint.keys():
#             model_1.load_state_dict(checkpoint['model_state_dict'], strict=False)
#         else:
#             model_1.load_state_dict(checkpoint, strict=False)
#         self.model_1 = model_1

#         # second model
#         model_2 = resnext50_32x4d(pretrained=True, progress=True)
#         # model_2.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
#         model_2.fc = nn.Identity()

#         lay4 = list(model_2.layer4)
#         channel0 = ChannelAttention(2048, 16)
#         spatial0 = SpatialAttention()
#         channel1 = ChannelAttention(2048, 16)
#         spatial1 = SpatialAttention()
#         channel2 = ChannelAttention(2048, 16)
#         spatial2 = SpatialAttention()
#         layer4_new = [lay4[0], channel0, spatial0, lay4[1], channel1, spatial1, lay4[2], channel2, spatial2]
#         model_2.layer4 = nn.Sequential(*layer4_new)

#         ckpt_2 = torch.load(
#             '/content/drive/MyDrive/Best_ResNext_cbam_map_92.75/ckpt34_acc_0.8830441236495972.pth')
#         model_2.load_state_dict(ckpt_2['model'], strict=False)
#         logger.info('acc of the second model is {}.'.format(ckpt_2['acc']))

#         # newmodel = torch.nn.Sequential(*(list(model_2.children())[:-1]))
#         # print(newmodel)
#         # self.model_2 = newmodel
#         self.model_2 = model_2
        
#         self.last_fc_1 = nn.Linear(2048 + 768, 500, True)
#         self.relu_last = nn.Sigmoid()
#         self.last_fc_2 = nn.Linear(500, 40, True)

#         # self.last_fc_1 = nn.Linear(2048 + 768, 512, True)
#         # # try dropout
#         # self.relu_last = nn.Sigmoid()
#         # self.last_fc_2 = nn.Linear(512, 40, True)


#     def forward(self, x1, x2):
#         y1, _ = self.model_1(x1)
#         y2 = self.model_2(x2)
#         # logger.info(y1.shape)
#         # logger.info(y2.shape)
#         y = torch.cat((y1, y2.squeeze()), dim=1)
#         # y = self.classifier(self.relu_last(y))
#         y = self.last_fc_2(self.relu_last(self.last_fc_1(y)))
#         # y = self.last_fc_2(self.last_fc_1(y))

#         # y = torch.stack((y1, y2), dim = 1).mean(dim = 1)
#         return y

class EnsembleModel_resnext_vit(nn.Module):
    def __init__(self, args, config, img_size=224, zero_head=True, num_classes=40):
        super().__init__()
        self.k = 2
        # first model
        model_1 = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
        checkpoint_file = args.input_dir
        checkpoint = torch.load(checkpoint_file)
        if 'model_state_dict' in checkpoint.keys():
            model_1.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model_1.load_state_dict(checkpoint, strict=False)
        self.model_1 = model_1

        # second model
        model_2 = resnext50_32x4d(pretrained=True, progress=True)
        model_2.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
        # model_2.fc = nn.Identity()

        lay4 = list(model_2.layer4)
        channel0 = ChannelAttention(2048, 16)
        spatial0 = SpatialAttention()
        channel1 = ChannelAttention(2048, 16)
        spatial1 = SpatialAttention()
        channel2 = ChannelAttention(2048, 16)
        spatial2 = SpatialAttention()
        layer4_new = [lay4[0], channel0, spatial0, lay4[1], channel1, spatial1, lay4[2], channel2, spatial2]
        model_2.layer4 = nn.Sequential(*layer4_new)

        ckpt_2 = torch.load(
            '/content/drive/MyDrive/Best_ResNext_cbam_map_92.75/ckpt34_acc_0.8830441236495972.pth')
        model_2.load_state_dict(ckpt_2['model'], strict=False)
        logger.info('acc of the second model is {}.'.format(ckpt_2['acc']))

        # newmodel = torch.nn.Sequential(*(list(model_2.children())[:-1]))
        # print(newmodel)
        # self.model_2 = newmodel
        self.model_2 = model_2
        
        # self.last_fc_1 = nn.Linear(2048 + 768, 500, True)
        # self.relu_last = nn.Sigmoid()
        # self.last_fc_2 = nn.Linear(500, 40, True)

        # self.last_fc_1 = nn.Linear(2048 + 768, 512, True)
        # # try dropout
        # self.relu_last = nn.Sigmoid()
        # self.last_fc_2 = nn.Linear(512, 40, True)


    def forward(self, x1, x2):
        y1, _ = self.model_1(x1)
        y2 = self.model_2(x2)
        # y = torch.cat((y1, y2.squeeze()), dim=1)
        # y = self.last_fc_2(self.relu_last(self.last_fc_1(y)))

        y = torch.stack((y1, y2), dim = 1).mean(dim = 1)
        return y

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def accuracy_classification(output, target):
    num_batch = output.shape[0]
    if not num_batch == target.shape[0]:
        raise ValueError
    pred = torch.argmax(output, dim=1)
    true_ = (pred == target).sum()
    # acc = (pred == target).mean()
    return true_ / num_batch


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def save_model_complete(args, model, optimizer, accuracy=None, step=0):
    if not accuracy:
        checkpoint_file = os.path.join(args.output_dir_every_checkpoint, "step_{}_checkpoint.pth".format(step))
    else:
        checkpoint_file = os.path.join(args.output_dir, "best_acc_step_{}_acc_{}_checkpoint.pth".format(step, accuracy))
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': accuracy,
    }, checkpoint_file)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "stanford40":
        num_classes = 40
    else:
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    checkpoint_file = args.input_dir
    checkpoint = torch.load(checkpoint_file)
    if 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # global_step = checkpoint['step'] + 1
        # best_acc = checkpoint['best_accuracy']
    else:
        model.load_state_dict(checkpoint)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def setup_for_ensemble(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "stanford40":
        num_classes = 40
    else:
        num_classes = 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    checkpoint_file = args.input_dir
    checkpoint = torch.load(checkpoint_file)
    if 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # global_step = checkpoint['step'] + 1
        # best_acc = checkpoint['best_accuracy']
    else:
        model.load_state_dict(checkpoint)
    # model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x1, x2, y = batch
        with torch.no_grad():
            logits = model(x1, x2)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


# def train(args, model_teacher, model_student):
#     """ Train the model """
#     model_student.to(args.device)
#     if args.local_rank in [-1, 0]:
#         os.makedirs(args.output_dir, exist_ok=True)
#         writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

#     args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

#     # Prepare dataset
#     train_loader, test_loader = get_loader_KD(args)

#     # Prepare optimizer and scheduler
#     optimizer = torch.optim.Adam(model_student.parameters(),
#                                  lr=args.learning_rate,
#                                  weight_decay=args.weight_decay)

#     # checkpoint_file = args.student_input_dir
#     # checkpoint_continue = torch.load(checkpoint_file)
#     # optimizer.load_state_dict(checkpoint_continue['optimizer_state_dict'])
    

#     t_total = args.num_steps
#     if args.decay_type == "cosine":
#         scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
#     else:
#         scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

#     if args.fp16:
#         [model_teacher, model_student], optimizer = amp.initialize(models=[model_teacher, model_student],
#                                                                    optimizers=optimizer,
#                                                                    opt_level=args.fp16_opt_level)
#         amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

#     # Distributed training
#     if args.local_rank != -1:
#         model_student = DDP(model_student, message_size=250000000, gradient_predivide_factor=get_world_size())
#         model_teacher = DDP(model_teacher, message_size=250000000, gradient_predivide_factor=get_world_size())

#     # Train!
#     logger.info("***** Running training *****")
#     logger.info("  Total optimization steps = %d", args.num_steps)
#     logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
#     logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
#                 args.train_batch_size * args.gradient_accumulation_steps * (
#                     torch.distributed.get_world_size() if args.local_rank != -1 else 1))
#     logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

#     global_step, best_acc = 0, 0

#     model_student.zero_grad()
#     set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
#     losses = AverageMeter()
#     # acc_train = AverageMeter()
#     accuracy_train = torchmetrics.Accuracy().cuda()
#     accuracy_train_teacher = torchmetrics.Accuracy().cuda()
#     # global_step, best_acc = 0, 0
#     while True:
#         model_teacher.eval()
#         model_student.train()
#         epoch_iterator = tqdm(train_loader,
#                               desc="Training (X / X Steps) (loss=X.X) (accuracy=X.X)",
#                               bar_format="{l_bar}{r_bar}",
#                               dynamic_ncols=True,
#                               disable=args.local_rank not in [-1, 0])

#         for step, batch in enumerate(epoch_iterator):
#             batch = tuple(t.to(args.device) for t in batch)
#             x1, x2, y = batch

#             with torch.no_grad():
#                 output_teacher, _ = model_teacher(x1)
#             output_student = model_student(x2)
#             # loss = loss_fn_kd(output_student, y, output_teacher, 0.6, 10)
#             loss = loss_fn_kd(output_student, y, output_teacher, 0.6, 10)
#             # accuracy_train = accuracy_classification(output_student, y)
#             accuracy_train(output_student.softmax(dim=-1), y)
#             accuracy_train_teacher(output_teacher.softmax(dim=-1), y)

#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps
#                 accuracy_train = accuracy_train / args.gradient_accumulation_steps
#                 accuracy_train_teacher = accuracy_train_teacher / args.gradient_accumulation_steps

#             if args.fp16:
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#             else:
#                 loss.backward()

#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 losses.update(loss.item() * args.gradient_accumulation_steps)
#                 # acc_train.update(accuracy_train * args.gradient_accumulation_steps)
#                 acc_train = accuracy_train.compute()
#                 acc_train_teacher = accuracy_train_teacher.compute()
#                 if args.fp16:
#                     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
#                 else:
#                     torch.nn.utils.clip_grad_norm_(model_student.parameters(), args.max_grad_norm)
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 scheduler.step()
#                 global_step += 1

#                 # epoch_iterator.set_description(
#                 #     "Training (%d / %d Steps) (loss=%2.5f) (accuracy=%2.5f)" % (global_step, t_total, losses.val,
#                 #                                                                 acc_train.val)
#                 # )
#                 epoch_iterator.set_description(
#                     "Training (%d / %d Steps) (loss=%2.5f) (accuracy=%2.5f)(teacher_acc=%2.5f)" % (global_step, t_total, losses.val,
#                                                                                 acc_train.item(), acc_train_teacher.item())
#                 )
#                 if args.local_rank in [-1, 0]:
#                     writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
#                     # writer.add_scalar("train/acc", scalar_value=acc_train.val, global_step=global_step)
#                     writer.add_scalar("train/acc", scalar_value=acc_train.item(), global_step=global_step)
#                     writer.add_scalar("train/acc_teacher", scalar_value=acc_train_teacher.item(), global_step=global_step)
#                     writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

#                 # save_checkpoint
#                 # save_model_complete(args, model, optimizer, accuracy = None, step = global_step)

#                 # if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
#                 if global_step % (4000 / 8) == 0 and args.local_rank in [-1, 0]:
#                     # logger.info("Train Accuracy: %2.5f" % acc_train.val)
#                     # logger.info("Train loss: %2.5f" % losses.val)
#                     accuracy = valid(args, model_student, writer, test_loader, global_step)

#                     if best_acc < accuracy:
#                         save_model_complete(args, model_student, optimizer, accuracy, global_step)
#                         best_acc = accuracy
#                     model_student.train()

#                 if global_step % t_total == 0:
#                     break
#         losses.reset()
#         if global_step % t_total == 0:
#             break

#     if args.local_rank in [-1, 0]:
#         writer.close()
#     logger.info("Best Accuracy: \t%f" % best_acc)
#     logger.info("End Training!")

def mAp(args, model, writer, test_loader, global_step):

    model.eval()
    class_acc = tnt.meter.APMeter()
    test_map = tnt.meter.mAPMeter()
    topacc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=False)
    conf_matrix = tnt.meter.ConfusionMeter(k=40,  normalized =False)
    class_acc.reset()
    test_map.reset()
    conf_matrix.reset()
    topacc.reset()

    with torch.no_grad():
        for x1, x2, y in test_loader:

            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)
            one_hot_y = torch.nn.functional.one_hot(y, num_classes=40)
            one_hot_y = one_hot_y.to(args.device)
            outputs = model(x1, x2)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            class_acc.add(probs, one_hot_y)
            test_map.add(probs, one_hot_y)
            conf_matrix.add(probs, one_hot_y)
            topacc.add(probs, y)
            
    logger.info('class accs are {}'.format(class_acc.value()))
    logger.info('mAp is equal to {}'.format(test_map.value()))
    logger.info('confusion matrix is {}'.format(conf_matrix.value()))
    logger.info('top 1th and 5th acc values are {}'.format(topacc.value()))

    writer.add_scalar("test/mAp", scalar_value=test_map.value(), global_step=global_step)

def train(args, model):
    """ Train the model """
    model.to(args.device)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader, class_names = get_loader_KD(args)


    # mAp
    # checkpoint_file = '/content/TrainedModels/best_acc_step_10500_acc_0.9152205350686913_checkpoint.pth'
    # checkpoint_continue = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint_continue['model_state_dict'], strict=True)
    # model.to(args.device)

    # model.eval()
    # # teacher34_model.eval()
    # class_acc = tnt.meter.APMeter()
    # test_map = tnt.meter.mAPMeter()
    # topacc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=False)
    # conf_matrix = tnt.meter.ConfusionMeter( k=40,  normalized =False)
    # class_acc.reset()
    # test_map.reset()
    # conf_matrix.reset()
    # topacc.reset()
    # with torch.no_grad():
    #     # for inputs, labels in dataloaders_dict['test']:
    #     for x1, x2, labels in test_loader:

    #         x1 = x1.to(args.device)
    #         x2 = x2.to(args.device)
    #         labels = labels.to(args.device)
    #         one_hot_labels = torch.nn.functional.one_hot(labels,num_classes=40)
    #         one_hot_labels = one_hot_labels.to(args.device)
    #         outputs = model(x1, x2)
    #         _, preds = torch.max(outputs, 1)
    #         probs = torch.nn.functional.softmax(outputs, dim=1)
    #         class_acc.add(probs, one_hot_labels)
    #         test_map.add(probs, one_hot_labels)
    #         conf_matrix.add(probs, one_hot_labels)
    #         topacc.add(probs, labels)
            
    # print('class accs are ', class_acc.value())
    # print('mAp is equal to ',test_map.value())
    # print('confusion matrix is ', conf_matrix.value())
    # print('top acc value is ', topacc.value())

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(model.parameters(),
    #                              lr=args.learning_rate,
    #                              weight_decay=args.weight_decay)

    # checkpoint_file = args.student_input_dir
    # checkpoint_continue = torch.load(checkpoint_file)
    # optimizer.load_state_dict(checkpoint_continue['optimizer_state_dict'])


    visualize_model(args, model, test_loader, class_names, num_images=1)
    

    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        # model_teacher = DDP(model_teacher, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step, best_acc = 0, 0

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    # acc_train = AverageMeter()
    accuracy_train = torchmetrics.Accuracy().cuda()
    accuracy_train_teacher = torchmetrics.Accuracy().cuda()
    loss_fct = nn.CrossEntropyLoss()
    mAp(args, model, writer, test_loader, -1)
    # global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X) (accuracy=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x1, x2, y = batch
            output = model(x1, x2)
            loss = loss_fct(output, y)

            accuracy_train(output.softmax(dim=-1), y)
            

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                accuracy_train = accuracy_train / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                # acc_train.update(accuracy_train * args.gradient_accumulation_steps)
                acc_train = accuracy_train.compute()
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) (accuracy=%2.5f)" % (global_step, t_total, losses.val,
                                                                                acc_train.item())
                )
                
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    # writer.add_scalar("train/acc", scalar_value=acc_train.val, global_step=global_step)
                    writer.add_scalar("train/acc", scalar_value=acc_train.item(), global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)

                # save_checkpoint
                # save_model_complete(args, model, optimizer, accuracy = None, step = global_step)

                # if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                if global_step % (4000 / 8) == 0 and args.local_rank in [-1, 0]:
                    # logger.info("Train Accuracy: %2.5f" % acc_train.val)
                    # logger.info("Train loss: %2.5f" % losses.val)
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    mAp(args, model, writer, test_loader, global_step)

                    if best_acc < accuracy:
                        save_model_complete(args, model, optimizer, accuracy, global_step)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


# def main():
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument("--name", required=True,
#                         help="Name of this run. Used for monitoring.")
#     parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stanford40"], default="cifar10",
#                         help="Which downstream task.")
#     parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
#                                                  "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
#                         default="ViT-B_16",
#                         help="Which variant to use.")
#     parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
#                         help="Where to search for pretrained ViT models.")
#     parser.add_argument("--output_dir", default="/content/TrainedModels", type=str,
#                         help="The output directory where checkpoints will be written.")
#     parser.add_argument("--output_dir_every_checkpoint",
#                         default="/content/", type=str,
#                         help="The output directory where checkpoints will be written.")
#     parser.add_argument("--input_dir",
#                         default="/content/drive/MyDrive/ViT_weights_layer11_to_end/best_acc_step_500_acc_0.9063629790310919_checkpoint.pth",
#                         type=str,
#                         help="The output directory where checkpoints will be written.")
#     # parser.add_argument("--student_input_dir",
#     #                     default="/content/drive/MyDrive/ResNeXT/Copy of ckpt30.pth",
#     #                     type=str,
#     #                     help="The output directory where checkpoints will be written.")
#     parser.add_argument("--student_input_dir",
#                         default="/content/drive/MyDrive/KD_First_weights/KD_ResNext_ViT_weights/ckpt1_acc_0.8617136478424072.pth",
#                         type=str,
#                         help="The output directory where checkpoints will be written.")
#     parser.add_argument("--img_size", default=224, type=int,
#                         help="Resolution size")
#     parser.add_argument("--train_batch_size", default=512, type=int,
#                         help="Total batch size for training.")
#     parser.add_argument("--eval_batch_size", default=64, type=int,
#                         help="Total batch size for eval.")
#     parser.add_argument("--eval_every", default=100, type=int,
#                         help="Run prediction on validation set every so many steps."
#                              "Will always run one evaluation at the end of training.")

#     # parser.add_argument("--learning_rate", default=3e-2, type=float,
#     #                     help="The initial learning rate for SGD.")
#     parser.add_argument("--learning_rate", default=1e-5, type=float,
#                         help="The initial learning rate for SGD.")
#     parser.add_argument("--weight_decay", default=0, type=float,
#                         help="Weight decay if we apply some.")
#     parser.add_argument("--num_steps", default=10000, type=int,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
#                         help="How to decay the learning rate.")
#     parser.add_argument("--warmup_steps", default=500, type=int,
#                         help="Step of training to perform learning rate warmup for.")
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")

#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="local_rank for distributed training on gpus")
#     parser.add_argument('--seed', type=int, default=42,
#                         help="random seed for initialization")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument('--fp16', action='store_true',
#                         help="Whether to use 16-bit float precision instead of 32-bit")
#     parser.add_argument('--fp16_opt_level', type=str, default='O2',
#                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#                              "See details at https://nvidia.github.io/apex/amp.html")
#     parser.add_argument('--loss_scale', type=float, default=0,
#                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
#                              "0 (default value): dynamic loss scaling.\n"
#                              "Positive power of 2: static loss scaling value.\n")
#     args = parser.parse_args()

#     # Setup CUDA, GPU & distributed training
#     if args.local_rank == -1:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         args.n_gpu = torch.cuda.device_count()
#     else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         torch.distributed.init_process_group(backend='nccl',
#                                              timeout=timedelta(minutes=60))
#         args.n_gpu = 1
#     args.device = device

#     # Setup logging
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#                         datefmt='%m/%d/%Y %H:%M:%S',
#                         level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
#     logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
#                    (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

#     # Set seed
#     set_seed(args)

#     # Model & Tokenizer Setup
#     args, model_teacher = setup(args)

#     # model_student = resnext50_32x4d(pretrained=True, progress=True)
#     # model_student.fc = nn.Linear(in_features=2048, out_features=40, bias=True)
#     # # student_checkpoint_file = args.student_input_dir
#     # # student_checkpoint = torch.load(student_checkpoint_file)
#     # # if 'model' in student_checkpoint.keys():
#     # #     model_student.load_state_dict(student_checkpoint['model'])
#     # # else:
#     # #     model_student.load_state_dict(student_checkpoint)
#     # lay4 = list(model_student.layer4)
#     # channel0_1 = ChannelAttention(2048, 16)
#     # spatial0_1 = SpatialAttention()
#     # channel1_1 = ChannelAttention(2048, 16)
#     # spatial1_1 = SpatialAttention()
#     # channel2_1 = ChannelAttention(2048, 16)
#     # spatial2_1 = SpatialAttention()
#     # layer4_new = [lay4[0], channel0_1, spatial0_1, lay4[1], channel1_1, spatial1_1,
#     #               lay4[2], channel2_1, spatial2_1]
#     # model_student.layer4 = nn.Sequential(*layer4_new)
#     # student_checkpoint_file = args.student_input_dir
#     # student_checkpoint = torch.load(student_checkpoint_file)
#     # if 'model_state_dict' in student_checkpoint.keys():
#     #     model_student.load_state_dict(student_checkpoint['model_state_dict'])
#     #     logger.info('loaded student weights from {}'.format(args.student_input_dir))
#     # elif 'model' in student_checkpoint.keys():
#     #     model_student.load_state_dict(student_checkpoint['model'])
#     #     logger.info('loaded student weights from {}'.format(args.student_input_dir))
#     # else:
#     #     model_student.load_state_dict(student_checkpoint)
#     #     logger.info('loaded student weights from {}'.format(args.student_input_dir))
#     model_student = EnsembleModel()
#     for name, param in model_student.named_parameters():
#       # if 'fc' in name or 'classifier' in name or 'senet' in name: #or 'layer3' in name:
#       #   logger.info(name)
#       #   param.requires_grad_(True)
#       # else:
#       #   param.requires_grad_(False)
#       param.requires_grad_(True)

#     # Training
#     train(args, model_teacher, model_student)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stanford40"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="/content/TrainedModels", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--output_dir_every_checkpoint",
                        default="/content/", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--input_dir",
                        default="/content/drive/MyDrive/vit_transforms_map_94.98/best_acc_step_300_acc_0.9101590744757773_checkpoint.pth",
                        type=str,
                        help="The output directory where checkpoints will be written.")
    # parser.add_argument("--student_input_dir",
    #                     default="/content/drive/MyDrive/ResNeXT/Copy of ckpt30.pth",
    #                     type=str,
    #                     help="The output directory where checkpoints will be written.")
    parser.add_argument("--student_input_dir",
                        default="/content/drive/MyDrive/KD_First_weights/KD_ResNext_ViT_weights/ckpt1_acc_0.8617136478424072.pth",
                        type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    # parser.add_argument("--learning_rate", default=3e-2, type=float,
    #                     help="The initial learning rate for SGD.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for SGD.")
    # parser.add_argument("--learning_rate", default=1e-4, type=float,
    #                     help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    # Prepare model

    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "stanford40":
        num_classes = 40
    else:
        num_classes = 100

    model = EnsembleModel_resnext_vit(args, config, args.img_size, zero_head=True, num_classes=num_classes)
    # checkpoint_file = '/content/drive/MyDrive/vit_res_ens_95.19_map/best_acc_step_10500_acc_0.9152205350686913_checkpoint.pth'
    # checkpoint = torch.load(checkpoint_file)
    # if 'model_state_dict' in checkpoint.keys():
    #     model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # else:
    #     model.load_state_dict(checkpoint, strict=True)

    # for name, param in model.named_parameters():
    #   param.requires_grad_(True)

    for name, param in model.named_parameters():
        # if 'last_fc' in name:
        #     param.requires_grad_(True)
        # elif 'model_2' in name:
        #     param.requires_grad_(False)
        # elif 'model_1' in name:
        #     param.requires_grad_(False)
        # else:
        #     param.requires_grad_(False)
        param.requires_grad_(True)
    
    # Training
    train(args, model)


if __name__ == "__main__":
    main()
