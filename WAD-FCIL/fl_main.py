from torchvision import datasets
from torch.utils.data import DataLoader
from WAD import WAD_model
from ResNet import *
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import *
from ProxyServer import *
from mini_imagenet import *
from tiny_imagenet import *
from option import args_parser
import torch.distributed as dist
import torchvision.models as models
import time
from cluster import cluster

# from TinyImagenet import TinyImageNet
import sys

def main():

    args = args_parser()
    #
    device_ids = list(map(int, args.device_ids.split(',')))
    # dist.init_process_group(backend='nccl')
    # device = torch.device(f'cuda:{device_ids[args.local_rank]}')
    # torch.cuda.set_device(device)
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    ## parameters for learning
    feature_extractor = resnet18_cbam(pretrained=False)  # feature extractor
    num_clients = args.num_clients
    old_client_0 = []
    old_client_1 = [i for i in range(args.num_clients)]  # [0, 1, 2,...29]
    new_client = []
    models = []

    ## seed settings
    setup_seed(args.seed)

    # print(torch.cuda.device_count())
    # torch.cuda.set_device(0)
    # ## model settings
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_g = network(args.numclass, feature_extractor)  # The model of the first round of tasks has 10 categories by default
    # model_old0 = network(args.numclass, feature_extractor)
    # model_new = network(args.numclass, feature_extractor)
    model_g = model_to_device(model_g, False, device, device_ids)
    # model_old0 = model_to_device(model_old0, False, device, device_ids)
    # model_new = model_to_device(model_new, False, device, device_ids)

    train_transform = transforms.Compose([
                                            # transforms.ToPILImage(),
                                            # transforms.Resize(64),
                                            transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ColorJitter(brightness=0.24705882352941178),
                                            transforms.RandomRotation(15),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                                        ])
    test_transform = transforms.Compose([
                                            # transforms.Resize(64),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                                        ])



    if args.dataset == 'cifar100':
        train_dataset = iCIFAR100('./dataset', transform=train_transform, download=False)
        test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=False)

    elif args.dataset == 'tiny_imagenet':
        # data_dir = './tiny-imagenet-200/'
        # train_dataset = TinyImageNet(data_dir, train=True, transform=train_transform)
        # test_dataset = TinyImageNet(data_dir, train=False, transform=test_transform)
        train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = train_dataset

    else:
        train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = train_dataset

    encode_model = LeNet(num_classes=100)

    for i in range(125):

        model_temp = WAD_model(i, args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                     args.epochs_local, args.learning_rate, train_dataset, device, encode_model, device_ids)
        # model_temp.model = model_to_device(model_temp.model, False, device, device_ids)
        models.append(model_temp)

        # print(model_temp.model.state_dict().keys())
        # sys.exit(0)
    # for i in range(125):
    #     print(id(models[i].model.feature))


    ## training log
    # output_dir = osp.join('./training_log', str(args.dataset) + '-' + str(args.epochs_global/args.tasks_global), str(args.e) + 'seed' + str(args.seed) + '_nomulti-model')
    output_dir = osp.join('./training_log', str(args.dataset) + '-' + str(args.epochs_global/args.tasks_global), str(args.e) + 'seed' + str(args.seed) + 'improve')
    # output_dir = osp.join('./training_log', str(args.dataset) + '-' + str(args.epochs_global/args.tasks_global), str(args.e) + 'seed' + str(args.seed) + '_noMWA')
    if not osp.exists(output_dir):
        os.system('mkdir -p ' + output_dir)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    out_file = open(osp.join(output_dir, 'log_tar_' + str(args.epochs_global/args.tasks_global) + '.txt'), 'w')
    log_str = 'method_{}, task_size_{}, learning_rate_{}, iid_level_{}, task_size_{}, temperature_{}'.format(args.method, args.task_size, args.learning_rate, args.iid_level, args.task_size, args.temperature)
    out_file.write(log_str + '\n')
    out_file.flush()

    classes_learned = args.task_size  # The number of categories for each round of task, which is 10 by default
    old_task_id = -1

    model_old_acc = 0.3
    model_old = [None for i in range(20)]
    log_end = ''
    acc_task = 0
    for ep_g in range(args.epochs_global):


        task_id = ep_g // args.tasks_global


        if task_id != old_task_id and old_task_id != -1:
            overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
            # Every time a new task comes, 10 new customers will be added
            new_client = [i for i in range(overall_client, overall_client + args.addclients)]

            # old_client_1 refer to customers who continue to receive new classes, taking 90% from all
            old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))

            old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
            num_clients = len(new_client) + len(old_client_1) + len(old_client_0)


        if ep_g % args.tasks_global == 1:
            exceptional_clients_index = cluster(models, num_clients)
            print('Abnormal clients are:：', exceptional_clients_index)



        if task_id != old_task_id and old_task_id != -1:
            # The second round of task starts to update the output layer of each client model (plus 10 logical outputs)
            classes_learned += args.task_size
            model_g = model_g.to('cpu')
            model_g.Incremental_learning(classes_learned)
            model_g = model_g.to(device)
            for model in models:
                model.model = copy.deepcopy(model_g)


        origin_clients_index = random.sample(range(num_clients), args.local_clients)  # 获得随机挑选的客户的索引
        if task_id > 0 and ep_g % args.tasks_global >= 1:
        # if 0:
            clients_index = [i for i in origin_clients_index if i not in exceptional_clients_index]
        else:
            clients_index = copy.deepcopy(origin_clients_index)
        # if args.local_rank == 0:
        print('The selected non abnormal local clients are:', clients_index)
        if len(clients_index) < 2:
            print('Skip this communication')
            continue

        clients_tasklenth = []
        w_local = []
        for c in clients_index:
            local_model = local_train(models, c, task_id, model_old, ep_g, old_client_0, ep_g, model_g)
            w_local.append(local_model)
            clients_tasklenth.append(len(models[c].task_id_list))



        if args.method == 'our':
            if ep_g % 10 not in [5, 7, 9]:
                model_for_m = weight_for_model(w_local=w_local, task_id=task_id, clients_tasklenth=clients_tasklenth, )
                model_global = FedAvg(model_for_m)
            else:
                model_global = FedAvg(w_local)
        else:
            model_global = FedAvg(w_local)

        model_g.load_state_dict(model_global)

        participant_exemplar_storing0(models, num_clients, model_g, old_client_0, task_id, clients_index)



        # Get the precision of the current global model on the test set of all existing classes
        acc_global = model_global_eval(model_g, test_dataset, task_id, args.task_size, device)
        log_str = 'Task: {}, Round: {} Accuracy = {:.3f}%'.format(task_id, ep_g, acc_global)

        if ep_g % args.tasks_global == args.tasks_global-1:
            log_end = log_end + 'Task: {}, Round: {} Accuracy = {:.3f}%\n'.format(task_id, ep_g, acc_global)
            acc_task += acc_global



        if ep_g % args.tasks_global == args.tasks_global-1:
            model_old[ep_g//args.tasks_global] = copy.deepcopy(model_g)
            model_old_acc = model_global_eval(model_old[ep_g//args.tasks_global], test_dataset, task_id, args.task_size, device)






        out_file.write(log_str + '\n')
        if ep_g % args.tasks_global == args.tasks_global-1:
            out_file.write(f'The accuracy of the old model is:{model_old_acc:.3f}' + '\n')
        if ep_g == args.epochs_global-1:
            out_file.write('\n' + log_end)
            out_file.write('\t\t\t\t\tAverage = {:.3f}%'.format(acc_task / (100/args.task_size)))
            # path = f'model_{args.method}.pth'
            # torch.save({'model_state_dict': model_g.state_dict()}, path)
        out_file.flush()

        print('commucation:%d，task:%d,acc:%.3f%% \n' % (ep_g, task_id, acc_global))


        # update id
        old_task_id = task_id
        # path = f'model_{args.method}.pth'
        # torch.save({'model_state_dict': model_g.state_dict()}, path)
if __name__ == '__main__':
    main()
