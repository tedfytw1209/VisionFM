# finetuning the weights of visionfm while training a multi-class classifier on top

import os
import argparse
import json
import copy
import torch
import torch.backends.cudnn as cudnn
import utils
import models
import numpy as np
import pandas as pd
from models.head import ClsHead
import wandb

from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score,matthews_corrcoef,
    multilabel_confusion_matrix,confusion_matrix
)
from collections import defaultdict


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

AD_LIST = ['control','mci','ad']
def select_n_slices(k,depth):
    if k<1:
        return int(k*depth/2)
    else:
        return k//2
class CSV_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,is_train,transfroms=[],k=0,class_to_idx={}, modality='OCT', patient_ids=None, pid_key = 'patient_id'):
        #common args
        self.transfroms = transfroms
        self.root_dir = img_dir
        self.loader = datasets.folder.default_loader
        data = pd.read_csv(csv_file)
        if not isinstance(is_train, list):
            is_train_l = [is_train]
        else:
            is_train_l = is_train
        is_train = is_train_l[0]
        if patient_ids is not None:
            self.annotations = data[data[pid_key].isin(patient_ids)]
            self.annotations['split'] = is_train
        elif 'split' in data.columns:
            self.annotations = data[data['split'].isin(is_train_l)]
        else:
            self.annotations = data
        print('Split: ', is_train_l,' Data len: ', self.annotations.shape[0])
        self.classes = [str(c) for c in self.annotations['label'].unique()]
        self.num_class = len(self.classes)
        #assert index order control, mci, ad
        if not class_to_idx:
            self.class_to_idx = {}
            i = 0
            for c in AD_LIST:
                if c in self.classes:
                    self.class_to_idx[c] = i
                    i+=1
            if i==0: #not in AD_LIST
                for c in sorted(self.classes):
                    self.class_to_idx[c] = int(c)
        else:
            self.class_to_idx = class_to_idx
        print('Class to idx: ', self.class_to_idx)
        self.channel = 3
        if modality == 'OCT' and 'OCT' in self.annotations.columns:
            image_names = self.annotations['OCT']
            print('OCT images: ', len(image_names))
            print('OCT image example: ', image_names.head())
        elif modality == 'CFP' and 'folder' in self.annotations.columns and 'fundus_imgname' in self.annotations.columns:
            image_names = self.annotations['folder'] + '/' + self.annotations['fundus_imgname']
            print('CFP images: ', len(image_names))
            print('CFP image example: ', image_names.head())
        else:
            print('Incompatible modality or missing columns: ', self.annotations.columns)
            image_names = self.annotations['image']
            print('Images: ', len(image_names))
        labels = self.annotations['label']
        #case for 2.5D
        if k>0:
            dcm_depths = self.annotations['depth']
            if is_train == 'train':
                samples = []
                for image_name,label,depth in zip(image_names, labels, dcm_depths):
                    med_point = (depth+1)//2
                    n_slice = select_n_slices(k,depth)
                    idx_start, idx_end = med_point-n_slice,med_point+n_slice+1
                    for i in range(idx_start,idx_end):
                        image_name = image_name.replace('_%d'%med_point,'_%d'%i) #!!! bug
                        if not image_name.endswith('.jpg'):
                            image_name = image_name + '.jpg'
                        samples.append((image_name, self.class_to_idx[str(label)]))
                self.half3D = False
            else:
                samples = []
                for image_name,label,depth in zip(image_names, labels, dcm_depths):
                    med_point = (depth+1)//2
                    n_slice = select_n_slices(k,depth)
                    idx_start, idx_end = med_point-n_slice,med_point+n_slice+1
                    image_name = image_name.replace('_%d'%med_point,'_%d'%i)
                    if not image_name.endswith('.jpg'):
                        image_name = image_name + '.jpg'
                    samples.append(([image_name for i in range(idx_start,idx_end)], self.class_to_idx[str(label)]))
                self.half3D = True
        else:
            samples = [(image_name if image_name.endswith('.jpg') else image_name + '.jpg', self.class_to_idx[str(label)]) for image_name,label in zip(image_names, labels)]
            self.half3D = False
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.k = k
        if k>0 and k<1:
            self.max_slice = int(((k*25) // 2)*2 + 1)
        elif k>1:
            self.max_slice = int((k//2) * 2 + 1)
        else:
            self.max_slice = 1
        self.modality = modality

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.half3D: #output multiple images
            img_name = [os.path.join(self.root_dir, each_name) for each_name in sample[0]]
            image = [self.loader(each_name) for each_name in img_name]
            image = [self.transfroms(each_image) for each_image in image]
            image_len = len(image)
            p3d = (0, 0, 0, 0 ,0 ,0 , 0, self.max_slice - image_len)
            image = torch.stack(image)
            image = torch.nn.functional.pad(image, p3d, mode='constant', value=0)
        else:
            img_name = os.path.join(self.root_dir, sample[0])
            image = self.loader(img_name)
            image = self.transfroms(image)
            image_len = 1
        
        label = int(sample[1])
        #debug visualization
        #print(image)
        return image, label

class RETFoundDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.data = []
        if 'PAPILA' in root:
            dr_folder_list = ['anormal', 'bsuspectglaucoma', 'cglaucoma']
        elif 'Glaucoma_fundus' in root:
            dr_folder_list = ['anormal_control', 'bearly_glaucoma', 'cadvanced_glaucoma']
        elif 'Retina' in root:
            dr_folder_list = ['anormal', 'bcataract', 'cglaucoma', 'ddretina_disease']
        elif 'OCTID' in root:
            dr_folder_list = ['ANormal', 'ARMD', 'CSR', 'Diabetic_retinopathy', 'Macular_Hole']
        elif 'JSIEC' in root:
            dr_folder_list = ['0.0.Normal',                                   '20.Massive hard exudates',
                    '0.1.Tessellated fundus',                       '21.Yellow-white spots-flecks',
                    '0.2.Large optic cup',                          '22.Cotton-wool spots',
                    '0.3.DR1',                                      '23.Vessel tortuosity',
                    '1.0.DR2',                                      '24.Chorioretinal atrophy-coloboma',
                    '1.1.DR3',                                      '25.Preretinal hemorrhage',
                    '10.0.Possible glaucoma',                       '26.Fibrosis',
                    '10.1.Optic atrophy',                           '27.Laser Spots',
                    '11.Severe hypertensive retinopathy',           '28.Silicon oil in eye',
                    '12.Disc swelling and elevation',               '29.0.Blur fundus without PDR',
                    '13.Dragged Disc',                              '29.1.Blur fundus with suspected PDR',
                    '14.Congenital disc abnormality',               '3.RAO',
                    '15.0.Retinitis pigmentosa',                    '4.Rhegmatogenous RD',
                    '15.1.Bietti crystalline dystrophy',            '5.0.CSCR',
                    '16.Peripheral retinal degeneration and break', '5.1.VKH disease',
                    '17.Myelinated nerve fiber',                    '6.Maculopathy',
                    '18.Vitreous particles',                        '7.ERM',
                    '19.Fundus neoplasm',                           '8.MH',
                    '2.0.BRVO',                                     '9.Pathological myopia',
                    '2.1.CRVO']
        elif 'IDRiD' in root:
            dr_folder_list = ['anoDR', 'bmildDR', 'cmoderateDR', 'dsevereDR', 'eproDR']
        else:
            dr_folder_list = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']

        for lbl, lbl_name in enumerate(dr_folder_list):
            img_files = os.listdir(os.path.join(root, split, lbl_name))
            for img_f in img_files:
                img_fpath = os.path.join(root, split, lbl_name, img_f)
                self.data.append({'img_fpath': img_fpath, 'label': lbl})
        self.transform = transform
        

    def __getitem__(self, index):
        entry = self.data[index]
        img = pil_loader(entry['img_fpath'])
        if self.transform is not None:
            img = self.transform(img)
        return img, entry['label']

    def __len__(self):
        return len(self.data)

#build dataset for select
def build_dataset(is_train, args, k=0, img_dir = '/orange/bianjiang/tienyu/OCT_AD/all_images/',transform=None, patient_ids=None, pid_key='patient_id'):
    if transform is None:
        transform = build_transform(is_train, args)
    
    if args.data_path.endswith('.csv'):
        dataset = CSV_Dataset(args.data_path, img_dir, is_train, transform, k, modality=args.modality, patient_ids=patient_ids, pid_key=pid_key)
    else:
        dataset = RETFoundDataset(root=args.data_path, split=is_train, transform=transform)

    return dataset

def build_transform(is_train, args):
    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")
    if not isinstance(is_train, list):
        is_train = [is_train]
    # train transform
    if 'train' in is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(args.input_size),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.RandomVerticalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean, std),
        ])
        return transform

    # eval transform
    transform = pth_transforms.Compose([
        pth_transforms.Resize(size=(args.input_size, args.input_size), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    return transform

def convert_to_one_hot(gts,num_classes):
    gts_one_hot = np.zeros((gts.shape[0], num_classes))
    for i in range(len(gts)):
        gts_one_hot[i][gts[i][0]] = 1

    return gts_one_hot


def eval_linear(args):
    wandb.init(
        project="VisionFM",
        name=args.task,
        config=args,
        dir=os.path.join('wandb_log',args.task),
    )
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    
    # fix the seed for reproducibility 
    utils.fix_random_seeds(args.seed)

    print(f"-------- Current Task: {args.task} Modality: {args.modality} -------")

    dataset_train = build_dataset(args=args, is_train='train',img_dir=args.img_dir)
    dataset_val = build_dataset(args=args, is_train='val', img_dir=args.img_dir)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = models.__dict__[args.arch](
        img_size = [args.input_size],
        patch_size=args.patch_size,
        num_classes=0,
        use_mean_pooling=args.avgpool_patchtokens==1)
    embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load visionfm pretrained weights
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    
    linear_classifier = ClsHead(embed_dim=embed_dim*4, num_classes=args.num_labels, layers=3)        
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': args.lr * 0.1 * (args.batch_size_per_gpu * utils.get_world_size()) / 256.}, {'params': linear_classifier.parameters()}],
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        betas=(0.9, 0.999), weight_decay=0.05
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_auc": 0.}
    start_epoch = to_restore["epoch"]
    best_auc = to_restore["best_auc"]
    aupr_with_best_auc = 0
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        linear_classifier.train()
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifier.eval()
            test_stats, output, target, output_labels = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)

            output = np.vstack(output)
            output_labels = np.concatenate(output_labels, axis=0)
            target = np.vstack(target)
            output_one_hot = convert_to_one_hot(output_labels,num_classes=args.num_labels)
            target_one_hot = convert_to_one_hot(target,num_classes=args.num_labels)
            target_1d = target.flatten()
            output_labels_1d = output_labels.flatten()

            auroc = roc_auc_score(target_one_hot, output, average='macro', multi_class='ovr')
            test_stats['auc'] = auroc
            aupr = average_precision_score(target_one_hot, output, average='macro')
            test_stats['aupr'] = aupr
            accuracy = accuracy_score(target_1d, output_labels_1d)
            hamming = hamming_loss(target_one_hot, output_one_hot)
            jaccard = jaccard_score(target_one_hot, output_one_hot, average='macro')
            average_precision = average_precision_score(target_one_hot, output_one_hot, average='macro')
            kappa = cohen_kappa_score(target_1d, output_labels_1d)
            f1 = f1_score(target_one_hot, output_one_hot, zero_division=0, average='macro')
            roc_auc = roc_auc_score(target_one_hot, output_one_hot, multi_class='ovr', average='macro')
            precision = precision_score(target_one_hot, output_one_hot, zero_division=0, average='macro')
            recall = recall_score(target_one_hot, output_one_hot, zero_division=0, average='macro')
            mcc = matthews_corrcoef(target_1d, output_labels_1d)
            add_dict ={
                'mcc': mcc,
                'accuracy': accuracy,
                'hamming': hamming,
                'jaccard': jaccard,
                'average_precision': average_precision,
                'kappa': kappa,
                'f1': f1,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall
            }
            test_stats.update(add_dict)

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()}}
        
            if utils.is_main_process() and (test_stats["auc"] >= best_auc):
                # always only save best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                log_stats["epoch"] = epoch
                wandb.log(log_stats, step=epoch)
                save_dict = {
                    "epoch": epoch + 1,
                    "classifier_state_dict": linear_classifier.state_dict(),
                    "visionfm_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_auc": test_stats["auc"],
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best_finetune.pth"))
                np.save(os.path.join(args.output_dir, 'best.npy'), output)
                np.save(os.path.join(args.output_dir, 'target.npy'), target)

                aupr_with_best_auc = test_stats['aupr']

            best_auc = max(best_auc, test_stats["auc"])
            print(f'Best val auc so far: {best_auc:.4f}; accompanying aupr: {aupr_with_best_auc:.4f}')
    wandb.log({
        "best_val_auc": best_auc,
        "best_val_aupr": aupr_with_best_auc
    })
    wandb.finish()
    print("Finetuning of VisionFM completed\n"
                "Best val auc: {acc:.4f}; accompanying aupr: {aupr:.4f}".format(acc=best_auc, aupr=aupr_with_best_auc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        intermediate_output = model.get_intermediate_layers(inp, n)
        if avgpool == 0:
            output = [x[:, 0] for x in intermediate_output] # only cls tokens
        elif avgpool == 1:
            output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)] # only patch tokens
        elif avgpool == 2:
            output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
        else:
            assert False, "Unkown avgpool type {}".format(avgpool)
        
        output = torch.cat(output, dim=-1)
        output = linear_classifier(output)

        # compute cross entropy loss
        num_class = output.shape[1]
        if num_class > 1: # multi-class case
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    model.eval()
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    targets, preds = [], []
    output_labels = []
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avgpool == 0:
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool == 1:
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool == 2:
                output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)
            
            output = torch.cat(output, dim=-1)
        
        output = linear_classifier(output)

        num_class = output.shape[1]
        if num_class > 1: # multi-class case
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())

        # save results
        if num_class > 1:  # multi-classes
            preds.append(output.softmax(dim=1).detach().cpu().numpy())
            output_label = output.argmax(dim=1)
            output_labels.append(np.expand_dims(output_label.detach().cpu().numpy(), axis=1))
            targets.append(np.expand_dims(target.detach().cpu().numpy(), axis=1))
        else:  # binary classification
            preds.append(output.detach().cpu().sigmoid().numpy())
            output_label = (output > 0.5).long().squeeze(dim=1)
            output_labels.append(np.expand_dims(output_label.detach().cpu().numpy(), axis=1))
            targets.append(np.expand_dims(target.detach().cpu().numpy(), axis=1))

        metric_logger.update(loss=loss.item())


    print('* val loss {losses.global_avg:.4f} '.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, preds, targets, output_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetuning VisionFM while training a multi-class classifier on top')
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.""")
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101', 'dalle_encoder'], help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='Input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of finetuning.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training the classifier""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--world_size", default=-1, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument('--data_path', default='/path/to/dataset/', type=str,
        help='Please specify path to the eye image data.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--modality', default='Fundus', type=str)
    parser.add_argument('--task', default='PAPILA', type=str)
    parser.add_argument('--extra', default='', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume finetuning')
    parser.add_argument('--img_dir', default='/orange/bianjiang/tienyu/OCT_AD/all_images/', type=str)
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Start finetuning {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
