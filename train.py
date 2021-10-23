from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta
import gc
from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dg import datasets
from dg import models
from dg.trainers import Trainer, DGTrainer
from dg.evaluators import Evaluator, extract_features
from dg.utils.data import IterLoader
from dg.utils.data import transforms as T
from dg.utils.data.sampler import RandomMultipleGallerySampler
from dg.utils.data.preprocessor import Preprocessor
from dg.utils.logging import Logger
from dg.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dg.utils.lr_scheduler import WarmupMultiStepLR

from dg.utils.faiss_rerank import compute_jaccard_distance

start_epoch = best_mAP = 0

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def main():
    args = parser.parse_args()
    main_worker(args)
    
    
def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load labeled synetic dataset")
    dataset_syne = get_data(args.dataset_syne, args.data_dir)
    print("==> Load unlabeled real-world dataset")
    dataset_real = get_data(args.dataset_real, args.data_dir)
    print("==> Load unseen target dataset")
    dataset_unseen = get_data(args.dataset_unseen, args.data_dir)
    '''
    train_loader_syne = get_train_loader(args, dataset_syne, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters)
    ''' 
    cluster_loader_real = get_test_loader(dataset_real, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset_real.train))
    test_loader_unseen = get_test_loader(dataset_unseen, args.height, args.width, args.batch_size, args.workers)
    
    syne_classes = dataset_syne.num_train_pids
    classes = syne_classes + len(dataset_real.train)
    

    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=classes)
    model.cuda()
    model = nn.DataParallel(model)
    
    # Load from checkpoint
    if args.resume:
        print("Load pretraining model.")
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))
    # Evaluator
    evaluator = Evaluator(model)
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if '_D' not in key:
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
    for epoch in range(start_epoch, args.epochs):
        print("Extract Real-world dataset features")
        real_features_d, _ = extract_features(model, cluster_loader_real, print_freq=50)
        real_features = torch.cat([real_features_d[f].unsqueeze(0) for f, _, _ in sorted(dataset_real.train)], 0)
        real_features = F.normalize(real_features, dim=1)
        rerank_dist = compute_jaccard_distance(real_features, k1=args.k1, k2=args.k2)
    
        if (epoch==start_epoch):
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)

        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
        
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset_real.train), cluster_id)):
                if id!=-1:
                    labels.append(syne_classes+id)
                else:
                    labels.append(syne_classes+num+outliers)
                    outliers += 1
            return torch.Tensor(labels).long()
        
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()
        
        R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
        assert((R_comp.min()>=0) and (R_comp.max()<=1))
        assert((R_indep.min()>=0) and (R_indep.max()<=1))

        cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
            cluster_R_comp[label.item()-syne_classes].append(comp.item())
            cluster_R_indep[label.item()-syne_classes].append(indep.item())
            cluster_img_num[label.item()-syne_classes]+=1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
        if (epoch==start_epoch):
            indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]
            
        pseudo_labeled_dataset = []
        outliers = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_real.train), pseudo_labels)):
            indep_score = cluster_R_indep[label.item()-syne_classes]
            comp_score = R_comp[i]
            if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()-syne_classes])):
                pseudo_labeled_dataset.append((fname,label.item(),cid))
            else:
                pseudo_labeled_dataset.append((fname,syne_classes+len(cluster_R_indep)+outliers,cid))
                pseudo_labels[i] = syne_classes+len(cluster_R_indep)+outliers
                outliers+=1
        
        
        #Refine
        stat = []
        for i in range(len(pseudo_labeled_dataset)):
            stat.append(pseudo_labeled_dataset[i][1])
        from collections import Counter
        stat_1 = dict(Counter(stat))
        
        dele = []
        for i in range(len(stat_1)):
            if(list(stat_1.values())[i]<5):
                dele.append(list(stat_1.keys())[i])
        pseudo_labeled_dataset_refine = []
        for i in range(len(pseudo_labeled_dataset)):
            if(pseudo_labeled_dataset[i][1] not in dele):
                pseudo_labeled_dataset_refine.append(pseudo_labeled_dataset[i])
                
        pseudo_labeled_dataset_relabel = []
        all_pids = {}
        for i in range(len(pseudo_labeled_dataset_refine)):
            pid = int(pseudo_labeled_dataset_refine[i][1])
            if (pid not in all_pids):
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]  + syne_classes  # relabel
            pseudo_labeled_dataset_relabel.append((pseudo_labeled_dataset_refine[i][0], pid, pseudo_labeled_dataset_refine[i][2]))
        
        assert len(pseudo_labeled_dataset_relabel)==len(pseudo_labeled_dataset_refine)
        
        print("The number of images before refinement:", len(pseudo_labeled_dataset))
        print("The number of images after refinement:", len(pseudo_labeled_dataset_refine))
        print("The number of labels before refinement:", len(stat_1))
        print("The number of labels after refinement:", len(stat_1) - len(dele))
        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))
        
        
        print("Setting Classifier Weights")
        labels = {}
        for i in range(len(pseudo_labeled_dataset_relabel)):
            label = pseudo_labeled_dataset_refine[i][1]
            if (label not in labels):
                labels[label] = [pseudo_labeled_dataset_refine[i][0]]
            else:
                labels[label].append(pseudo_labeled_dataset_refine[i][0])
        
        centers = torch.Tensor()
        for i in labels.keys():
            images = labels[i]
            for j in range(len(images)):
                images[j] = images[j].split('/')[-1]
            fs = []
            for name in real_features_d.keys():
                if(name.split('/')[-1] in images):
                    fs.append(real_features_d[name])
            assert len(fs)>=5
            center = sum(fs)/len(fs)
            centers = torch.cat((centers,center),-1)
        cluster_centers = centers.reshape(len(labels),-1)
        
        assert len(cluster_centers) == len(stat_1) - len(dele)
        
        model.module.classifier.weight.data[syne_classes:syne_classes + len(cluster_centers)].copy_(F.normalize(cluster_centers, dim=1).float().cuda())
        
        train_loader = get_train_loader(args, dataset_real, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_relabel + dataset_syne.train)
        #train_loader_syne.new_epoch()
        train_loader.new_epoch()

        
        num_class = syne_classes + (index2label>=5).sum() #index2label==1 should NOT be included
        print("The number of Class is", num_class)
        
        if(epoch>=30):
            params_G = []
            params_D = []
            for key, value in model.named_parameters():
                if not value.requires_grad:
                    continue
                if '_D' not in key:
                    params_G += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
                if '_D' in key:
                    params_D += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
            optimizer_G = torch.optim.Adam(params_G)
            optimizer_D = torch.optim.Adam(params_D)
            lr_scheduler_G = WarmupMultiStepLR(optimizer_G, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
            lr_scheduler_D = WarmupMultiStepLR(optimizer_D, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
            trainer = DGTrainer(model, num_class, syne_classes)
            trainer.train(epoch, train_loader, optimizer_G, optimizer_D, train_iters=len(train_loader), print_freq=args.print_freq) 
            lr_scheduler_G.step()
            lr_scheduler_D.step()
        
        else:
            trainer = Trainer(model, num_class)
            trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=2000)
            lr_scheduler.step()
        
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            _, mAP = evaluator.evaluate(test_loader_unseen, dataset_unseen.query, dataset_unseen.gallery, cmc_flag=True)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_%d.pth.tar'%(epoch)))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
         
        
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DomainMix: Learning Generalizable Person Re-Identification Without Human Annotations")
    # data
    parser.add_argument('-dsy', '--dataset-syne', type=str, default='randperson_subset',
                        choices=datasets.names())
    parser.add_argument('-dre', '--dataset-real', type=str, default='msmt17',
                        choices=datasets.names())
    parser.add_argument('-dun', '--dataset-unseen', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)#64
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20, 40], help='milestones for the learning rate decay')
    
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=100)
    
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()



