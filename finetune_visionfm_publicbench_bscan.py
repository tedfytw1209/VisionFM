# Finetuning VisionFM (OCT encoder) on OphFoundation's public OCT B-scan
# benchmark datasets (duke14, glaucoma, oimhs, umn).
#
# Data loading uses PublicOCTBscanDataset (dataset_public_oct.py), ported
# from MIRAGE's mutils/dataset_public_oct.py / run_cls_tuning_bscan.py --
# each dataset's raw volumes + fold-split CSV, middle-slice-per-volume,
# already verified working against these datasets on the Linux server.
# num_classes is auto-inferred from the train fold's label column, not
# hardcoded per dataset.

import os
import json
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms

import utils
import models
from models.head import ClsHead
from dataset_public_oct import PublicOCTBscanDataset

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score,
    cohen_kappa_score, matthews_corrcoef,
)


# Per-dataset raw image subdirectory (under --data_root) and fold-split CSV
# location (under --csv_root), per OphFoundation's
# finetune-UF-benchmark_*_single.sh reference scripts (same mapping as
# MIRAGE's run_cls_tuning_bscan.py).
PUBLIC_OCT_DATASETS = {
    'duke14': {
        'image_subdir': 'DUKE_14_Srin/duke14_processed',
        'csv_subdir': 'finetune_duke14_fewshot_3D_10folds_effective_fold',
        'csv_pattern': 'duke14_fold_split{fold}.csv',
    },
    'glaucoma': {
        'image_subdir': 'GLAUCOMA/glaucoma_processed',
        'csv_subdir': 'finetune_glaucoma_fewshot_3D_10folds_correct_visit',
        'csv_pattern': 'glaucoma_fold_{fold}_split.csv',
    },
    'oimhs': {
        'image_subdir': 'OIMHS_dataset/cls_images',
        'csv_subdir': 'finetune_oimhs_fewshot_3D_10folds_correct_',
        'csv_pattern': 'oimhs_fold_{fold}_split_ref.csv',
    },
    'umn': {
        'image_subdir': 'UMN/UMN_dataset/image_classification',
        'csv_subdir': 'finetune_umn_fewshot_3D_10folds_correct',
        'csv_pattern': 'umn_fold_{fold}_split_ref.csv',
    },
}


def resolve_paths(args):
    dataset_config = PUBLIC_OCT_DATASETS[args.data_set]
    args.image_root = os.path.join(args.data_root, dataset_config['image_subdir'])
    csv_dir = os.path.join(args.csv_root, dataset_config['csv_subdir'])
    args.csv_file = os.path.join(csv_dir, dataset_config['csv_pattern'].format(fold=args.fold))
    return args


def build_transform(subset, args):
    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")
    if subset == 'train':
        return pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(args.input_size),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.RandomVerticalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean, std),
        ])
    return pth_transforms.Compose([
        pth_transforms.Resize(size=(args.input_size, args.input_size), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])


def build_dataset(subset, args):
    return PublicOCTBscanDataset(
        csv_file=args.csv_file,
        root_dir=args.image_root,
        split=subset,
        dataset_name=args.data_set,
        transform=build_transform(subset, args),
    )


def convert_to_one_hot(gts, num_classes):
    gts_one_hot = np.zeros((gts.shape[0], num_classes))
    for i in range(len(gts)):
        gts_one_hot[i][gts[i][0]] = 1
    return gts_one_hot


def compute_metrics(preds, targets, output_labels, num_labels):
    output = np.vstack(preds)
    output_labels = np.concatenate(output_labels, axis=0)
    target = np.vstack(targets)
    output_one_hot = convert_to_one_hot(output_labels, num_classes=num_labels)
    target_one_hot = convert_to_one_hot(target, num_classes=num_labels)
    target_1d = target.flatten()
    output_labels_1d = output_labels.flatten()

    return {
        'auc': roc_auc_score(target_one_hot, output, average='macro', multi_class='ovr'),
        'aupr': average_precision_score(target_one_hot, output, average='macro'),
        'accuracy': accuracy_score(target_1d, output_labels_1d),
        'hamming': hamming_loss(target_one_hot, output_one_hot),
        'jaccard': jaccard_score(target_one_hot, output_one_hot, average='macro'),
        'kappa': cohen_kappa_score(target_1d, output_labels_1d),
        'f1': f1_score(target_one_hot, output_one_hot, zero_division=0, average='macro'),
        'precision': precision_score(target_one_hot, output_one_hot, zero_division=0, average='macro'),
        'recall': recall_score(target_one_hot, output_one_hot, zero_division=0, average='macro'),
        'mcc': matthews_corrcoef(target_1d, output_labels_1d),
    }


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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

        loss = nn.CrossEntropyLoss()(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    model.eval()
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    targets, preds, output_labels = [], [], []
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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

        loss = nn.CrossEntropyLoss()(output, target)

        preds.append(output.softmax(dim=1).detach().cpu().numpy())
        output_label = output.argmax(dim=1)
        output_labels.append(np.expand_dims(output_label.detach().cpu().numpy(), axis=1))
        targets.append(np.expand_dims(target.detach().cpu().numpy(), axis=1))

        metric_logger.update(loss=loss.item())

    print('* val loss {losses.global_avg:.4f} '.format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, preds, targets, output_labels


def eval_linear(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)

    args = resolve_paths(args)
    print(f"-------- Current Task: {args.task} Dataset: {args.data_set} Fold: {args.fold} Modality: {args.modality} -------")

    dataset_train = build_dataset('train', args)
    args.num_labels = len(set(dataset_train.targets))
    print(f'Auto-detected {args.num_labels} classes from {args.csv_file}')
    dataset_val = build_dataset('val', args)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers,
        pin_memory=True, shuffle=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    model = models.__dict__[args.arch](
        img_size=[args.input_size], patch_size=args.patch_size, num_classes=0,
        use_mean_pooling=args.avgpool_patchtokens == 1)
    embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    linear_classifier = ClsHead(embed_dim=embed_dim * 4, num_classes=args.num_labels, layers=3)
    linear_classifier = linear_classifier.cuda()

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': args.lr * 0.1 * (args.batch_size_per_gpu * utils.get_world_size()) / 256.},
         {'params': linear_classifier.parameters()}],
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        betas=(0.9, 0.999), weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    best_auc = 0.
    aupr_with_best_auc = 0.
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        linear_classifier.train()
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifier.eval()
            val_stats, preds, targets, output_labels = validate_network(
                val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            val_stats.update(compute_metrics(preds, targets, output_labels, args.num_labels))

            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}

            if val_stats['auc'] >= best_auc:
                with (Path(args.output_dir) / 'log.txt').open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')
                save_dict = {
                    'epoch': epoch + 1,
                    'classifier_state_dict': linear_classifier.state_dict(),
                    'visionfm_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_auc': val_stats['auc'],
                    'num_labels': args.num_labels,
                }
                torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint_best_finetune.pth'))
                aupr_with_best_auc = val_stats['aupr']

            best_auc = max(best_auc, val_stats['auc'])
            print(f'Best val auc so far: {best_auc:.4f}; accompanying aupr: {aupr_with_best_auc:.4f}')

    print("Finetuning of VisionFM completed\n"
          "Best val auc: {acc:.4f}; accompanying aupr: {aupr:.4f}".format(acc=best_auc, aupr=aupr_with_best_auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetuning VisionFM on a public OCT B-scan benchmark dataset')
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="Whether or not to use global average pooled features or the [CLS] token.")
    parser.add_argument('--arch', default='vit_base', type=str, choices=[
        'vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'swin_tiny', 'swin_small',
        'swin_base', 'swin_large', 'resnet50', 'resnet101', 'dalle_encoder'], help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='Input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, required=True,
        help='Path to pretrained VisionFM weights (OCT encoder).')
    parser.add_argument('--checkpoint_key', default='teacher', type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of finetuning.')
    parser.add_argument('--lr', default=0.001, type=float,
        help='Learning rate at the beginning of training the classifier')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument('--local_rank', default=0, type=int, help='Please ignore and do not set this argument.')
    parser.add_argument('--data_root', type=str, required=True,
        help='Root directory containing the raw per-dataset OCT volume/slice image folders'
            ' (e.g. .../OCTCubeM/assets/ext_oph_datasets/).')
    parser.add_argument('--csv_root', type=str, required=True,
        help='Root directory containing the per-dataset fold-split CSV directories'
            ' (e.g. .../OphFoundation/Public_OCT_split/).')
    parser.add_argument('--data_set', type=str, required=True, choices=sorted(PUBLIC_OCT_DATASETS.keys()),
        help='Dataset name.')
    parser.add_argument('--fold', default=0, type=int, help='Which pre-computed fold-split CSV to use (0-9).')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--modality', default='OCT', type=str)
    parser.add_argument('--task', default='duke14', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help='Epoch frequency for validation.')
    parser.add_argument('--output_dir', default='.', help='Path to save logs and checkpoints')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
