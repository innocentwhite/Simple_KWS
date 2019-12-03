import argparse
import os
import logging
import time
import datetime
from time import strftime
import sys
import uuid
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from model import models
import input_data
from input_data import GSCDataset

import tensorflow as tf
import numpy as np

from adamW import AdamW
import admm

from utils import *

from config import Config

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--arch', '-a', type=str,
                    help='What model architecture to use')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_size_info', type=int, nargs="+",
                    help='Model dimensions - different for various models')

best_prec1 = 0

def main():
    global config, best_prec1
    args = parser.parse_known_args()[0].__dict__
    config = Config()
    for key, value in args.items():
        if value is not None:
            setattr(config, key, value)
    if config.logger:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        try:
            os.makedirs("logger", exist_ok=True)
        except TypeError:
            raise Exception("Direction not create!")
        logger.addHandler(
            logging.FileHandler(strftime('logger/GSC_%m-%d-%Y-%H:%M_id_') + str(uuid.uuid4()) + '.log', 'a'))
        global print
        print = logger.info

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Check the save_dir exists or not
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    print("Current network is {}".format(config.arch))

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()
    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(config.wanted_words.split(','))),
        config.sample_rate, config.clip_duration_ms, config.window_size_ms,
        config.window_stride_ms, config.dct_coefficient_count)

    print(model_settings)

    time_shift_samples = int((config.time_shift_ms * config.sample_rate) / 1000)

    train_loader = torch.utils.data.DataLoader(
        GSCDataset(config.data_url, config.data_dir, config.silence_percentage, config.unknown_percentage,
                   config.wanted_words.split(','), config.validation_percentage, config.testing_percentage,
                   model_settings, sess, config.arch, mode="training", background_frequency=config.background_frequency,
                   background_volume_range=config.background_frequency, time_shift=time_shift_samples), shuffle=True,
        batch_size=config.batch_size, num_workers=config.workers)
    print("train set size: {}".format(len(train_loader.dataset)))
    val_loader = torch.utils.data.DataLoader(
        GSCDataset(config.data_url, config.data_dir, config.silence_percentage, config.unknown_percentage,
                   config.wanted_words.split(','), config.validation_percentage, config.testing_percentage,
                   model_settings, sess, config.arch, mode="validation"), batch_size=config.batch_size,
        num_workers=config.workers)
    print("validation set size: {}".format(len(val_loader.dataset)))
    test_loader = torch.utils.data.DataLoader(
        GSCDataset(config.data_url, config.data_dir, config.silence_percentage, config.unknown_percentage,
                   config.wanted_words.split(','), config.validation_percentage, config.testing_percentage,
                   model_settings, sess, config.arch, mode="testing"), batch_size=config.batch_size,
        num_workers=config.workers)
    print("test set size: {}".format(len(test_loader.dataset)))

    model = models.create_model(config, model_settings)
    model.cuda()
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Trying load with dict 'state_dict'")
                # try:
                # if not config.admm_quant:
                #     model.set_alpha(checkpoint['alpha'])
                
                model.load_state_dict(checkpoint['state_dict'])
                # except:
                #     print("Cann't load model")
                #     return

        else:
            print("=> no checkpoint found at '{}'".format(config.resume))
            return

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if config.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=True)
    elif config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), config.lr,
                                    weight_decay=config.weight_decay)
    elif config.optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), config.lr,
                                        weight_decay=config.weight_decay)
    elif config.optimizer_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), config.lr,
                                        weight_decay=config.weight_decay)
    elif config.optimizer_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), config.lr,
                                        weight_decay=config.weight_decay)
    else:
        raise ValueError("The optimizer type is not defined!")

    if config.evaluate:
        # validate(val_loader, model, criterion)
        val_acc = validate_by_step(config, val_loader, model, criterion, model_settings, sess)
        # test(test_loader, model, criterion)
        test_acc = test_by_step(config, test_loader, model, criterion, model_settings, sess)
        return

    if config.admm_quant:
        pass
        # name_list = []

        # for name, w in model.named_parameters():
        #     if "weight" or "bias" in name:
        #         name_list.append(name)

        # print("Quantized Layer name list is :")
        # print(", ".join(name_list))

        # print("Before quantized:")

        # validate_by_step(config, audio_processor, model, criterion, model_settings, sess)

        # admm.admm_initialization(config, model, device, name_list, print)
        # print("After quantized:")
        # validate_quant_by_step(config, audio_processor, model, criterion, model_settings,
        #                        sess, name_list, device)

        # for epoch in range(config.start_epoch, config.epochs):

        #     if config.lr_scheduler == 'default':
        #         adjust_learning_rate(optimizer, epoch)

        #     elif config.lr_scheduler == 'cosine':
        #         pass

        #     admm_quant_train_by_step(config, audio_processor, model, criterion, optimizer, epoch, model_settings,
        #                              time_shift_samples, sess, name_list, device)

        #     # evaluate on validation set
        #     print("After Quantized:")
        #     prec1, quantized_model = validate_quant_by_step(config, audio_processor, model, criterion, model_settings,
        #                                                     sess, name_list, device)

        #     # remember best prec@1 and save checkpoint
        #     is_best = prec1 > best_prec1
        #     if is_best:
        #         path_name = os.path.join(config.save_dir,
        #                                  '{arch}_{type}_{num_bits}bits_quantized_GSC_acc_{prec1:.3f}.pt'.format(
        #                                      arch=config.arch, 
        #                                      type=config.quant_type, num_bits=config.num_bits,
        #                                      prec1=best_prec1))
        #         new_path_name = os.path.join(config.save_dir,
        #                                      '{arch}_{type}_{num_bits}bits_quantized_GSC_acc_{prec1:.3f}.pt'.format(
        #                                          arch=config.arch, type=config.quant_type, num_bits=config.num_bits,
        #                                          prec1=prec1))
        #         if os.path.isfile(path_name):
        #             os.remove(path_name)

        #         best_prec1 = prec1
        #         save_checkpoint(quantized_model, new_path_name)
        #         print("Admm training, best top 1 acc {best_prec1:.3f}".format(best_prec1=best_prec1))
        #         print("Best testing dataset:")
        #         test_by_step(config, audio_processor, quantized_model, criterion, model_settings, sess)
        #     else:
        #         print("Admm training, best top 1 acc {best_prec1:.3f}, current top 1 acc {prec1:.3f}".format(
        #             best_prec1=best_prec1, prec1=prec1))


    else:

        for epoch in range(0, config.epochs):

            if config.lr_scheduler == 'default':
                adjust_learning_rate(optimizer, epoch)
            elif config.lr_scheduler == 'cosine':
                pass
                # scheduler.step()

            # train for one epoch
            train_by_step(config, train_loader, model, criterion, optimizer, epoch, model_settings, time_shift_samples,
                          sess)

            # evaluate on validation set
            # prec1 = validate(val_loader, model, criterion)
            prec1 = validate_by_step(config, val_loader, model, criterion, model_settings, sess)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                path_name = os.path.join(config.save_dir,
                                         '{arch}_GSC_acc_{prec1:.3f}.pt'.format(
                                             arch=config.arch, prec1=best_prec1))
                new_path_name = os.path.join(config.save_dir,
                                             '{arch}_GSC_acc_{prec1:.3f}.pt'.format(
                                                 arch=config.arch, prec1=prec1))
                if os.path.isfile(path_name):
                    os.remove(path_name)
                best_prec1 = prec1
                save_checkpoint(model, new_path_name)
                print("Current best validation accuracy {best_prec1:.3f}".format(best_prec1=best_prec1))
            else:
                print("Current validation accuracy {prec1:.3f}, "
                      "best validation accuracy {best_prec1:.3f}".format(prec1=prec1, best_prec1=best_prec1))

        # test(test_loader, model, criterion)
        test_by_step(config, test_loader, model, criterion, model_settings, sess)


def train_by_step(config, train_loader, model, criterion, optimizer, epoch, model_settings, time_shift_samples, sess):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    train_set_size = len(train_loader.dataset)
    max_step_epoch = train_set_size // config.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.Tensor(target.float()).cuda()
        _, target = target.max(dim=1)
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input.float()).cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output = output.double
        # loss = loss.double()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, max_step_epoch, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def admm_quant_train_by_step(config, audio_processor, model, criterion, optimizer, epoch, model_settings,
                             time_shift_samples, sess, name_list, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    mixed_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    train_set_size = audio_processor.set_size('training')
    max_step_epoch = train_set_size // config.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i in range(0, train_set_size, config.batch_size):
        input, target = audio_processor.get_data(
            config.batch_size, 0, model_settings, config.background_frequency,
            config.background_volume, time_shift_samples, 'training', sess)

        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        ce_loss = criterion(output, target_var)
        admm.z_u_update(config, model, device, epoch, i, name_list, print)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(model, ce_loss)  # append admm losss

        # compute gradient
        optimizer.zero_grad()
        mixed_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        ce_losses.update(ce_loss.data, input.size(0))
        mixed_losses.update(mixed_loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i // config.batch_size) % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Cross Entropy Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                  'Mixed Loss {mixed_loss.val:.4f} ({mixed_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i // config.batch_size, max_step_epoch, batch_time=batch_time,
                data_time=data_time, ce_loss=ce_losses, mixed_loss=mixed_losses, top1=top1))


def validate_quant_by_step(config, audio_processor, model, criterion, model_settings, sess, name_list, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    quantized_model = models.create_model(config, model_settings)
    quantized_model.alpha = model.alpha
    quantized_model.Q = model.Q
    quantized_model.Z = model.Z
    quantized_model.load_state_dict(model.state_dict())
    quantized_model.cuda()
    admm.apply_quantization(config, quantized_model, name_list, device)

    quantized_model.eval()
    valid_set_size = audio_processor.set_size('validation')
    max_step_epoch = valid_set_size // config.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i in range(0, valid_set_size, config.batch_size):
        input, target = audio_processor.get_data(config.batch_size, i, model_settings, 0.0,
                                                 0.0, 0, 'validation', sess)
        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)
        target = target.cuda()
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        output = quantized_model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i // config.batch_size) % config.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i // config.batch_size, max_step_epoch, batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, quantized_model


def validate_by_step(config, val_loader, model, criterion, model_settings, sess):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    valid_set_size = len(val_loader.dataset)
    max_step_epoch = valid_set_size // config.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    miss_count = 0
    far_count = 0
    false_count = 0
    frr_count = 0
    outputs = torch.tensor([])
    targets = []
    i = 0
    for input, target in val_loader:
        i+=1
        target = torch.Tensor(target.float()).cuda()
        _, target = target.max(dim=1)
        target = target.cuda()
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input.float()).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        output = model(input_var)

        loss = criterion(output, target_var)
        
        output = output.float()
        loss = loss.float()

        if outputs.shape[0] == 0:
            outputs = output
        else:
            outputs = torch.cat([outputs, output], 0)
        targets = targets + target.tolist()
        miss_t, false_t, frr_divider, far_divider = evaluate(output.data, target.tolist())
        
        miss_count += miss_t
        frr_count += frr_divider
        false_count += false_t
        far_count += far_divider

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            false_reject_rate = miss_count / frr_count
            false_alarm_rate = false_count / far_count
            print('Validation: [{0}/{1}]\t'
                  'frr {false_reject_rate:.2f}%\t'
                  'far {false_alarm_rate:.2f}%\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, max_step_epoch, false_reject_rate=false_reject_rate*100, false_alarm_rate=false_alarm_rate*100,
                top1=top1))
    # if (config.evaluate):
    #     ROC(outputs, targets)
    # import pickle
    # with open('outputs_circuit.pkl', 'wb') as f:
    #     pickle.dump(outputs, f)
    # with open('targets_circuit.pkl', 'wb') as f:
    #     pickle.dump(targets, f)
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def test_by_step(config, test_loader, model, criterion, model_settings, sess):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    test_set_size = len(test_loader.dataset)
    max_step_epoch = test_set_size // config.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    i = 0 
    for input, target in test_loader:
        i+=1
        target = torch.Tensor(target.float()).cuda()
        _, target = target.max(dim=1)
        target = target.cuda()
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input.float()).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, max_step_epoch, batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def test(test_loader, model, criterion):
    """
    Run test evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        _, target = target.max(dim=1)
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(model, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    state = {}
    state['state_dict'] = model.state_dict()
    if config.admm_quant:
        state['alpha'] = model.alpha

    torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs"""
    lr = config.lr * (0.5 ** (epoch // 20))
    print("learning rate ={}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
