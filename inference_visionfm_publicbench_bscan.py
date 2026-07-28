# Evaluating a VisionFM checkpoint (from finetune_visionfm_publicbench_bscan.py)
# on OphFoundation's public OCT B-scan benchmark datasets (duke14, glaucoma,
# oimhs, umn).
#
# NOTE: OphFoundation's own fold-split CSVs for these 4 datasets never
# populate split=='test' -- their reference script
# (finetune-UF-benchmark_*_single.sh / pytorch_image_classification_our_model-v2-UF.py)
# reports 'val' under the name "test" instead of a true held-out split.
# MIRAGE's run_cls_tuning_bscan.py does the same for comparability; this
# script follows suit rather than evaluating against an always-empty split.

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


def build_transform(args):
    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")
    return pth_transforms.Compose([
        pth_transforms.Resize(size=(args.input_size, args.input_size), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])


def build_dataset(args):
    # See module docstring: 'val' stands in for a held-out test split here.
    return PublicOCTBscanDataset(
        csv_file=args.csv_file,
        root_dir=args.image_root,
        split='val',
        dataset_name=args.data_set,
        transform=build_transform(args),
    )


def convert_to_one_hot(gts, num_classes):
    gts_one_hot = np.zeros((gts.shape[0], num_classes))
    for i in range(len(gts)):
        gts_one_hot[i][gts[i][0]] = 1
    return gts_one_hot


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    model.eval()
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
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

    print('* test loss {losses.global_avg:.4f} '.format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, preds, targets, output_labels


def eval_linear(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    utils.fix_random_seeds(args.seed)

    args = resolve_paths(args)
    print(f"-------- Current Task: {args.task} Dataset: {args.data_set} Fold: {args.fold} Modality: {args.modality} -------")

    dataset_test = build_dataset(args)
    args.num_labels = len(set(dataset_test.targets))
    print(f'Auto-detected {args.num_labels} classes from {args.csv_file}')

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers,
        pin_memory=True, shuffle=False,
    )
    print(f"Data loaded with {len(dataset_test)} test imgs.")

    model = models.__dict__[args.arch](
        img_size=[args.input_size], patch_size=args.patch_size, num_classes=0,
        use_mean_pooling=args.avgpool_patchtokens == 1)
    embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    linear_classifier = ClsHead(embed_dim=embed_dim * 4, num_classes=args.num_labels, layers=3)
    linear_classifier = linear_classifier.cuda()

    state_dict = torch.load(args.pretrained_weights, map_location='cpu')
    classifier_state_dict = state_dict.get('classifier_state_dict', None)
    if classifier_state_dict is None:
        print("Cannot find the weights for classifier (decoder). Please refer to our fine-tuning instruction!!")
        print("The classifier would utlize the random weights!!")
    else:
        msg = linear_classifier.load_state_dict(classifier_state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    model.eval()
    linear_classifier.eval()
    test_stats, preds, targets, output_labels = validate_network(
        test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)

    output = np.vstack(preds)
    output_labels = np.concatenate(output_labels, axis=0)
    target = np.vstack(targets)
    output_one_hot = convert_to_one_hot(output_labels, num_classes=args.num_labels)
    target_one_hot = convert_to_one_hot(target, num_classes=args.num_labels)
    target_1d = target.flatten()
    output_labels_1d = output_labels.flatten()

    auroc = roc_auc_score(target_one_hot, output, average='macro', multi_class='ovr')
    test_stats['auc'] = auroc
    aupr = average_precision_score(target_one_hot, output, average='macro')
    test_stats['aupr'] = aupr
    test_stats['accuracy'] = accuracy_score(target_1d, output_labels_1d)
    test_stats['hamming'] = hamming_loss(target_one_hot, output_one_hot)
    test_stats['jaccard'] = jaccard_score(target_one_hot, output_one_hot, average='macro')
    test_stats['kappa'] = cohen_kappa_score(target_1d, output_labels_1d)
    test_stats['f1'] = f1_score(target_one_hot, output_one_hot, zero_division=0, average='macro')
    test_stats['precision'] = precision_score(target_one_hot, output_one_hot, zero_division=0, average='macro')
    test_stats['recall'] = recall_score(target_one_hot, output_one_hot, zero_division=0, average='macro')
    test_stats['mcc'] = matthews_corrcoef(target_1d, output_labels_1d)

    print(f"AUC: {auroc}, AUPR: {aupr}")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'best.npy'), output)
    np.save(os.path.join(args.output_dir, 'target.npy'), target)
    scalar_stats = {k: float(v) for k, v in test_stats.items() if k != 'epoch'}
    with open(os.path.join(args.output_dir, 'test_stats.json'), 'w') as f:
        json.dump(scalar_stats, f, indent=2)
    print(json.dumps(scalar_stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluating a VisionFM public OCT B-scan checkpoint')
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
        help='Checkpoint saved by finetune_visionfm_publicbench_bscan.py.')
    parser.add_argument('--checkpoint_key', default='visionfm_state_dict', type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument('--local_rank', default=0, type=int, help='Please ignore and do not set this argument.')
    parser.add_argument('--data_root', type=str, required=True,
        help='Root directory containing the raw per-dataset OCT volume/slice image folders.')
    parser.add_argument('--csv_root', type=str, required=True,
        help='Root directory containing the per-dataset fold-split CSV directories.')
    parser.add_argument('--data_set', type=str, required=True, choices=sorted(PUBLIC_OCT_DATASETS.keys()),
        help='Dataset name.')
    parser.add_argument('--fold', default=0, type=int, help='Which pre-computed fold-split CSV to use (0-9).')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--modality', default='OCT', type=str)
    parser.add_argument('--task', default='duke14', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default='.', help='Path to save logs and checkpoints')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
