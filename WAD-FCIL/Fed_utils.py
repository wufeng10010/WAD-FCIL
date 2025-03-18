import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random
from option import args_parser
import copy
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os.path as osp


args = args_parser()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def model_to_device(model, parallel, device, device_ids):
    if parallel:
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]],
                                                    output_device=device_ids[args.local_rank],
                                                    find_unused_parameters=True)
    else:
        model = model.to(device)

    return model


def update_task_id(clients, num, old_client0, task_id,):
    for index in range(num):
        if index not in old_client0:
            clients[index].task_id_list.append(task_id)


def participant_exemplar_storing(clients, num, old_client0, task_id, clients_index,
                                 ):
    for index in range(num):
        if index not in clients_index:
            if index in old_client0:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()

        clients[index].signal = False

def participant_exemplar_storing0(clients, num, model_g, old_client0, task_id, clients_index,):
    for index in range(num):
        clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client0:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()

        clients[index].signal = False


def local_train(clients, index, task_id, model_old, ep_g, old_client, epg, model_g):
    clients[index].model = copy.deepcopy(model_g)
    clients[index].epg = epg
    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)

    clients[index].update_new_set()

    clients[index].train(ep_g, model_old, index)
    if isinstance(clients[index].model, torch.nn.parallel.DistributedDataParallel):
        local_model = clients[index].model.module.state_dict()
    else:
        local_model = clients[index].model.state_dict()

    if args.local_rank == 0:
        print('*' * 60)

    return copy.deepcopy(local_model)


def local_train_and_load(clients, index, model_g, task_id, model_old, ep_g, old_client, epg):
    clients[index].model = copy.deepcopy(model_g)
    clients[index].epg = epg
    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)

    clients[index].update_new_set()
    clients[index].train(ep_g, model_old, index)
    if isinstance(clients[index].model, torch.nn.parallel.DistributedDataParallel):
        local_model = clients[index].model.module.state_dict()
    else:
        local_model = clients[index].model.state_dict()

    if args.local_rank == 0:
        print('*' * 60)

    return copy.deepcopy(local_model)  # proto_grad


def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg


def model_global_eval(model_g, test_dataset, task_id, task_size, device):
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)

    accuracy = 100 * correct / total
    model_g.train()
    return accuracy


def model_global_eval1(model_g, test_dataset, task_id, task_size, device, epg):
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)
    correct, total = 0, 0
    np_predict = np.array([])
    np_label   = np.array([])
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
        np_predict = np.concatenate((np_predict, predicts.cpu().numpy()))
        np_label = np.concatenate((np_label, labels.cpu().numpy()))
        confusion_marix(np_predict, np_label, epg)


    accuracy = 100 * correct / total
    model_g.train()
    return accuracy


def confusion_marix(np_predict, np_label, epg):
    cm = confusion_matrix(np_label, np_predict)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='coolwarm', fmt='g')
    plt.xticks(np.arange(0, 100, step=20), [f'class{i}' for i in np.arange(0, 100, step=20)])
    plt.yticks(np.arange(0, 100, step=20), [f'class{i}' for i in np.arange(0, 100, step=20)])
    plt.tick_params(axis='x', which='both', pad=15)
    plt.tick_params(axis='y', which='both', pad=15)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    output_dir = f'confusion_matrix_{args.method}'
    if not osp.exists(output_dir):
        os.system('mkdir -p ' + output_dir)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(f'confusion_matrix_{args.method}/epg{epg}.png')


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1:
        dist = torch.norm(w_1[key].cpu() - w_2[key].cpu())
        dist_total += dist.cpu()

    return dist_total.cpu().item()


def cluster_FedAvg(models, w_radio):
    w_avg = copy.deepcopy(models[0])
    for i in range(0, len(models)):
        if i == 0:
            for k in w_avg.keys():
                w_avg[k] = models[i][k] * w_radio[i]

        else:
            for k in w_avg.keys():
                w_avg[k] += models[i][k] * w_radio[i]

    return w_avg


def clu2ster_FedAvg(models, w_radio):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():

        for i in range(0, len(models)):
            if i == 0:
                w_avg[k] = models[i][k] * w_radio[i]

            else:
                w_avg[k] += models[i][k] * w_radio[i]

    return w_avg

def weight_for_model(w_local, task_id, clients_tasklenth, ):
    n = [500, 400, 300, 200, 100, 100, 100, 1 / 10, 1 / 10, 1 / 1000,
         1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000, 1 / 1000
         ]
    model_for_m = []
    for m in range(6):
        select_k_clents = random.sample(range(len(w_local)), len(w_local)-2)

        w_select_local = []
        select_length = []
        for i in select_k_clents:
            w_select_local.append(w_local[i])
            select_length.append(clients_tasklenth[i])
        w_avg = FedAvg(w_select_local)

        dist_list = []
        for i in range(len(select_k_clents)):
            dist = model_dist(w_select_local[i], w_avg)
            dist_list.append(dist)

        w = [(select_length[i]/(task_id+1))  *  (1-np.exp(-dist_list[i] / args.a)) for i in range(len(select_k_clents))]

        totol = sum(w)
        w_for_eachclient = [w[i] / totol for i in range(len(w))]

        w_g_new = cluster_FedAvg(w_select_local, w_for_eachclient)
        model_for_m.append(w_g_new)
    return model_for_m


def migrate(clients, clients_index, image_groups, device):
    clients_index = sorted(clients_index)

    models_list = []
    for i in clients_index:
        models_list.append(copy.deepcopy(clients[i].model))

    logit_for_each_client = []
    for i in range(len(clients_index)):
        logit_output = []
        for group in image_groups:
            inputs = torch.stack(group).squeeze()
            inputs = inputs.cuda(device)
            with torch.no_grad():
                outputs = torch.mean(clients[i].model(inputs), dim=0).squeeze()
            # 处理模型的输出结果
            logit_output.append(outputs)
        logit_for_each_client.append(logit_output)

    related_matrix = [[] for _ in range(len(clients_index))]
    for client_student in range(len(clients_index)):
        for client_teacher in range(len(clients_index)):
            if client_student == client_teacher:
                related_matrix[client_student].append(float('inf'))
            else:
                student = logit_for_each_client[client_student]
                teacher = logit_for_each_client[client_teacher]
                related_number = 0
                for c in range(10):
                    KL = nn.KLDivLoss(reduction='batchmean')
                    related_number += abs(float(KL(torch.log(torch.sigmoid(student[c])), torch.sigmoid(teacher[c])).cpu()))
                related_matrix[client_student].append(related_number)

    for client in range(len(clients_index)):
        min_index = related_matrix[client].index(min(related_matrix[client]))
        clients[clients_index[client]].model = copy.deepcopy(models_list[min_index])


def migrate_based_taskid(models, num_clients, clinets_index, task_id):
    for index in range(num_clients):
    # for model in models:
        model_list = models[index].task_id_list
        task_different_mount_list = []
        for client in clinets_index:
            if index != client:
                client_task_list = models[client].task_id_list
                a = similarity_digree(model_list, client_task_list)
                task_different_mount_list.append(a)
        indices = find_max_indices(task_different_mount_list)
        # if len(indices) == 1:
        #     models[index].model = copy.deepcopy(models[clinets_index[indices[0]]].model)
        if len(indices) > 5:
            local_model = []
            for ind in indices:
                local_model.append(copy.deepcopy(models[clinets_index[ind]].model.state_dict()))
            if index in clinets_index:
                local_model.append(copy.deepcopy(models[index].model.state_dict()))
            model_for_m = weight_for_model(local_model, task_id)
            model_exchange = FedAvg(model_for_m)
            models[index].model.load_state_dict(model_exchange)
        else:
            local_model = []
            for ind in indices:
                local_model.append(copy.deepcopy(models[clinets_index[ind]].model.state_dict()))
            if index in clinets_index:
                local_model.append(copy.deepcopy(models[index].model.state_dict()))
            model_exchange = FedAvg(local_model)
            models[index].model.load_state_dict(model_exchange)



def similarity_digree(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection_length = len(set1.intersection(set2))
    return intersection_length


def find_max_indices(lst):
    max_value = max(lst)
    max_indices = [i for i, value in enumerate(lst) if value == max_value]  # 使用 enumerate() 函数获取索引值
    return max_indices
