from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth, SoftTripletLoss
from .utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.num_classes = num_classes

    def train(self, epoch, data_loader, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader.next()
            
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out, _ = self.model(s_inputs)
            s_cls_out = s_cls_out[:,:self.num_classes]
            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = 10 * loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

class DGTrainer(object):
    def __init__(self, model, num_classes, syne_classes, margin=0.0):
        super(DGTrainer, self).__init__()
        self.model = model
        self.criterion_ce_c = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_ce_d = CrossEntropyLabelSmooth(num_classes=2, epsilon=0).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.num_classes = num_classes
        self.syne_classes = syne_classes

    def train(self, epoch, data_loader, optimizer_G, optimizer_D, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce_c = AverageMeter()
        losses_ce_d = AverageMeter()
        losses_tr = AverageMeter()
        losses_mix = AverageMeter()
        precisions_c = AverageMeter()
        precisions_d = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader.next()
            data_time.update(time.time() - end)

            s_inputs, targets_c = self._parse_data(source_inputs)
            s_features, s_cls_out_c, s_cls_out_d = self.model(s_inputs)
            s_cls_out_c = s_cls_out_c[:,:self.num_classes]
            
            targets_d = (targets_c > self.syne_classes).type_as(targets_c)
            
            # backward main #
            loss_ce_c, loss_ce_d, loss_tr, prec1_c, prec1_d = self._forward(s_features, s_cls_out_c, s_cls_out_d, targets_c, targets_d)
            
            
            s_cls_out_d_p = torch.softmax(s_cls_out_d,1)
            loss_mix = torch.mean(torch.sum(s_cls_out_d_p*torch.log(s_cls_out_d_p) + 0.5, dim=1))
            
            
            if (i%10==0):
                loss = loss_ce_d
                optimizer_D.zero_grad()
                loss.backward()
                optimizer_D.step()
            else:
                loss = 1 * loss_ce_c + loss_tr + 1 * loss_mix
                optimizer_G.zero_grad()
                loss.backward()
                optimizer_G.step()
                
                
                
            
            
            losses_ce_c.update(loss_ce_c.item())
            losses_ce_d.update(loss_ce_d.item())
            losses_tr.update(loss_tr.item())
            losses_mix.update(loss_mix.item())
            precisions_c.update(prec1_c)
            precisions_d.update(prec1_d)

            

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce_c {:.3f} ({:.3f})\t'
                      'Loss_ce_d {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Loss_mix {:.3f} ({:.3f})\t'
                      'Prec_c {:.2%} ({:.2%})\t'
                      'Prec_d {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce_c.val, losses_ce_c.avg,
                              losses_ce_d.val, losses_ce_d.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_mix.val, losses_mix.avg,
                              precisions_c.val, precisions_c.avg,
                              precisions_d.val, precisions_d.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs_c, s_outputs_d, targets_c, targets_d):
        loss_ce_c = self.criterion_ce_c(s_outputs_c, targets_c)
        loss_ce_d = self.criterion_ce_d(s_outputs_d, targets_d)
        loss_tr = self.criterion_triple(s_features, s_features, targets_c)
        prec_c, = accuracy(s_outputs_c.data, targets_c.data)
        prec_c = prec_c[0]
        prec_d, = accuracy(s_outputs_d.data, targets_d.data)
        prec_d = prec_d[0]

        return loss_ce_c, loss_ce_d, loss_tr, prec_c, prec_d




