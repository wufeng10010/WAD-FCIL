import copy

import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from option import args_parser

from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader, DistributedSampler
import random
import time
from Fed_utils import *
from collections import defaultdict


args = args_parser()


def get_one_hot(target, num_class, device):
    target = target.to(device)
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class WAD_model:

    def __init__(self, client_id, numclass, feature_extractor, batch_size, task_size, memory_size, epochs,
                 learning_rate, train_set, device, encode_model, device_ids):

        super(WAD_model, self).__init__()
        self.client_id = client_id
        self.device_ids = device_ids
        self.epg = 0
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.encode_model = encode_model

        self.index = None
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = 0
        self.learned_numclass = 0
        self.learned_classes = []
        self.transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None
        self.old_model0 = None
        self.old_model1 = None
        self.train_dataset = train_set
        self.start = True
        self.signal = False

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1
        self.device = device
        self.last_entropy = 0
        self.task_id_list = []
        self.distill_p = [None, 1.0, 1.3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        self.loss      = [None, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0]
        self.temp     = [5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.temp2     = [5.0, 4.9, 4.8, 4.7, 4.6, 4.4, 4.3, 4.2, 4.1, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                          4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

    # get incremental train data
    def beforeTrain(self, task_id_new, group):
        if task_id_new != self.task_id_old:
            if args.local_rank == 0 and group != 0:
                self.signal = True
            self.task_id_old = task_id_new
            self.numclass = self.task_size * (task_id_new + 1)
            if group != 0:
                self.task_id_list.append(task_id_new)
                if self.current_class != None:
                    self.last_class = self.current_class
                self.current_class = random.sample([x for x in range(self.numclass - self.task_size, self.numclass)],
                                                   args.iid_level)
                # print(self.current_class)
            else:
                self.last_class = None

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, False)

    def update_new_set(self):

        self.model.eval()

        if self.signal and (self.last_class != None):
            self.learned_numclass += len(self.last_class)
            self.learned_classes += self.last_class

            m = int(self.memory_size / self.learned_numclass)
            self._reduce_exemplar_sets(m)

            for i in self.last_class:
                images = self.train_dataset.get_image_class(i)
                # print(images.shape)
                self._construct_exemplar_set(images, m)
        self.model.train()

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, True)

    def _get_train_and_test_dataloader(self, train_classes, mix):
        if mix:
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.learned_classes)
        else:
            self.train_dataset.getTrainData(train_classes, [], [])

        # self.train_sampler = DistributedSampler(self.train_dataset)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  # sampler=self.train_sampler,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=8,
                                  pin_memory=True)

        return train_loader

    # train model
    def train(self, ep_g, model_old, index):

        # self.old_model = model_to_device(self.old_model, False, self.device, self.device_ids)

        # self.model.train()
        self.epg = ep_g
        self.index = index
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        self.old_model = model_old

        if self.old_model[0] != None:
            for oldmodel in self.old_model:
                if oldmodel != None:
                    oldmodel = model_to_device(oldmodel, False, self.device, self.device_ids)
                    oldmodel.eval()

        time_start = time.time()
        for epoch in range(20):
            running_loss = 0

            loss_cur_sum, loss_mmd_sum = [], []
            if (epoch + ep_g * 20) % 200 == 100:  # epoch=0，ep_g=5, 15, 25, 35... 95
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 5
            elif (epoch + ep_g * 20) % 200 == 150:  # epoch=10, ep_g=7，17, 27... 97
                if self.numclass > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 25
                else:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)

            elif (epoch + ep_g * 20) % 200 == 180:  # epoch=0, ep_g=9, 19, 29... 99
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 125

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                loss_value = self._compute_loss(indexs, images, target)

                opt.zero_grad()
                loss_value.backward()
                opt.step()
                running_loss += loss_value.item()


        time_end = time.time()  # 结束计时
        t = time_end - time_start
        if args.local_rank == 0:
            print(
                f'client{self.index},{self.epg}communication,{self.task_id_old}轮task, time:{t}s',
                'LR: {:0.6f}'.format(opt.param_groups[0]['lr']))

    def entropy_signal(self, loader):
        self.model.eval()
        start_ent = True
        res = False

        for step, (indexs, imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = entropy(softmax_out)

            if start_ent:
                all_ent = ent.float().cpu()
                all_label = labels.long().cpu()
                start_ent = False
            else:
                all_ent = torch.cat((all_ent, ent.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.long().cpu()), 0)
        overall_avg = torch.mean(all_ent).item()
        if args.local_rank == 0:
            print(overall_avg)
        if overall_avg - self.last_entropy > 1.2:
            res = True

        self.last_entropy = overall_avg

        self.model.train()

        return res

    # 获得损失
    def _compute_loss(self, indexs, imgs, labels):


        if self.task_id_old == 0:
            output = self.model(imgs)
            target = get_one_hot(labels, self.numclass, self.device)
            output, target = output.to(self.device), target.to(self.device)

            # w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))  # 原文中的Lgc
            # loss_cur = criterion(output, label)

            return 1.0 * loss_cur

        else:
            if args.method == 'glfc' or args.method == 'our-mmd':
                output = self.model(imgs)
                target = get_one_hot(labels, self.numclass, self.device)
                output, target = output.to(self.device), target.to(self.device)
                loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
                distill_target = target.clone()
                with torch.no_grad():
                    old_target = torch.sigmoid(self.old_model[self.task_id_old - 1](imgs))
                old_task_size = old_target.shape[1]
                distill_target[..., :old_task_size] = old_target
                loss_old = F.binary_cross_entropy_with_logits(output, distill_target)
                return 1.0* loss_cur + 1.0 * loss_old


            elif args.method == 'LGA':
                output = self.model(imgs)
                target = get_one_hot(labels, self.numclass, self.device)
                output, target = output.to(self.device), target.to(self.device)
                w = self.efficient_old_class_weight(output, labels)
                loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))
                distill_target = target.clone()

                with torch.no_grad():
                    old_target = torch.sigmoid(self.old_model[self.task_id_old - 1](imgs))

                old_task_size = old_target.shape[1]
                distill_target[..., :old_task_size] = old_target
                loss_old = torch.mean(w * F.binary_cross_entropy_with_logits(output, distill_target, reduction='none'))

                return 1.0 * loss_cur + 1.0 * loss_old



            elif args.method == 'icarl':
                output = self.model(imgs)
                target = get_one_hot(labels, self.numclass, self.device)
                output, target = output.to(self.device), target.to(self.device)
                loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
                distill_target = get_one_hot(labels, self.numclass-self.task_size, self.device)

                loss_old = F.binary_cross_entropy_with_logits(output[..., :self.numclass-self.task_size], distill_target)
                return 0.5 * loss_cur + 0.5 * loss_old

            else:
                if 0 < self.task_id_old < (args.epochs_global // args.tasks_global) // 2:
                    output = self.model(imgs)
                    target = get_one_hot(labels, self.numclass, self.device)
                    output, target = output.to(self.device), target.to(self.device)

                    loss_cur = torch.mean(
                        F.binary_cross_entropy_with_logits(output, target, reduction='none'))

                    distill_target = target.clone()
                    with torch.no_grad():
                        old_target = torch.sigmoid(self.old_model[self.task_id_old - 1](imgs))
                    old_task_size = old_target.shape[1]
                    distill_target[..., :old_task_size] = old_target

                    soft_loss = nn.KLDivLoss(reduction='batchmean')
                    loss_old = soft_loss(torch.log(torch.sigmoid(output / args.temperature)),
                                         distill_target / args.temperature)
                    return 1.0 * loss_cur + 1.0 * loss_old

                else:
                    output = self.model(imgs)
                    target = get_one_hot(labels, self.numclass, self.device)
                    output, target = output.to(self.device), target.to(self.device)

                    newlabels = torch.masked_select(labels, labels >= args.task_size * self.task_id_old)
                    newtarget = get_one_hot(newlabels, self.numclass, self.device)
                    newtarget = newtarget.to(self.device)
                    identical_tasklabel_to_images = defaultdict(list)

                    distill_target = None
                    images = None

                    for img, label in zip(imgs, labels):
                        identical_tasklabel_to_images[label.item() // args.task_size].append(img)

                    for tasklabel in identical_tasklabel_to_images.keys():
                        if tasklabel < self.task_id_old:
                            with torch.no_grad():
                                tasklabel_distilltarget = torch.sigmoid(
                                    self.old_model[tasklabel](torch.stack(identical_tasklabel_to_images[tasklabel])))
                            padded_tensor = F.pad(tasklabel_distilltarget,
                                                  (0, self.numclass - tasklabel_distilltarget.shape[1]), value=0.)

                            if distill_target is None:
                                distill_target = padded_tensor
                            else:
                                distill_target = torch.cat((distill_target, padded_tensor), dim=0)

                        else:
                            with torch.no_grad():
                                tasklabel_distilltarget = torch.sigmoid(self.old_model[tasklabel - 1](
                                    torch.stack(identical_tasklabel_to_images[tasklabel])))
                            # padded_tensor = F.pad(tasklabel_distilltarget, (0, self.numclass - tasklabel_distilltarget.shape[1]), value=0.)

                            old_task_size = tasklabel_distilltarget.shape[1]
                            newtarget[..., :old_task_size] = tasklabel_distilltarget

                            if distill_target is None:
                                distill_target = newtarget
                            else:
                                distill_target = torch.cat((distill_target, newtarget), dim=0)

                        if images is None:
                            images = torch.stack(identical_tasklabel_to_images[tasklabel])
                        else:
                            images = torch.cat((images, torch.stack(identical_tasklabel_to_images[tasklabel])), dim=0)

                    images = images.to(self.device)
                    distill_output = self.model(images)

                    loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))

                    soft_loss = nn.KLDivLoss(reduction='batchmean')
                    loss_old = soft_loss(torch.log(torch.sigmoid(distill_output / args.temperature)),
                                         distill_target / args.temperature)
                    return 1.0 * loss_cur + 1.0 * loss_old

    # weight on LGA
    def efficient_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        N, C = pred.size(0), pred.size(1)
        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = get_one_hot(label, self.numclass, self.device)
        g = torch.abs(pred.detach() - target)
        if self.task_id_old > 0:
            z = torch.div(self.task_id_old, self.task_id_old + 1)
            z = g.clone().fill_(z)
            g = torch.pow(g, z)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.learned_classes) != 0:
            for i in self.learned_classes:
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))
            index2 = torch.ne(ids, -1).float()
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)
            ids = label.view(-1, 1)
            for i in range(self.task_id_old):
                task = (i + 1) * self.task_size
                classes = [n for n in self.learned_classes if n >= task - self.task_size and n < task]
                for j in classes:
                    ids = torch.where(ids != j, ids, ids.clone().fill_(-1))
                index = torch.eq(ids, -1).float()
                if index.sum() != 0:
                    w = torch.div(g * index, (g * index).sum() / index.sum())
                else:
                    w = g.clone().fill_(0.)
                w2 += w
            w2 = torch.clamp(w2, 0.5, 5)
            return w2

        else:
            w = g.clone().fill_(1.)
        return w

    # weight on GLFC
    def efficient_old_class_weight_GLFC(self, output, label):

        label = label.to(self.device)
        pred = torch.sigmoid(output)

        N, C = pred.size(0), pred.size(1)

        class_mask = pred.data.new(N, C).fill_(0)  #
        ids = label.view(-1, 1)
        class_mask.scatter_(1, label.long().view(-1, 1), 1.)

        target = get_one_hot(label, self.numclass, self.device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.learned_classes) != 0:
            for i in self.learned_classes:
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))

            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)

            w = w1 + w2

        else:
            w = g.clone().fill_(1.)

        return w

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_set.append(exemplar)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):

        x = self.Image_transform(images, transform).to(self.device)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            feature_extractor_output = F.normalize(self.model.module.feature_extractor(x).detach()).cpu().numpy()
        else:
            feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def _KD_loss(pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]