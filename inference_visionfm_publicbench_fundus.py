# Evaluating a VisionFM checkpoint (from finetune_visionfm_publicbench_fundus.py)
# on the held-out test split of a public fundus-photo benchmark dataset.
#
# Data loading is ImageFolder-based. The checkpoint's training class mapping
# must match all train/val/test folder mappings before the held-out test split
# is evaluated, avoiding the hardcoded class-folder-name list used by
# RETFoundDataset in inference_visionfm_for_multiclass_classification_UF.py.

import os
import json
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import models
from models.head import ClsHead
from publicbench_fundus_protocol import (
    compute_classification_metrics,
    validate_imagefolder_splits,
    validate_split_class_maps,
)


def build_transform(args):
    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")
    return pth_transforms.Compose([
        pth_transforms.Resize(size=(args.input_size, args.input_size), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])


def build_dataset(subset, args):
    root = os.path.join(args.data_path, subset)
    dataset = datasets.ImageFolder(root, transform=build_transform(args))
    print(f'{subset} class_to_idx: {dataset.class_to_idx}')
    return dataset


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

    print(f"-------- Current Task: {args.task} Modality: {args.modality} -------")

    checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
    checkpoint_class_to_idx = checkpoint.get('class_to_idx')
    if checkpoint_class_to_idx is None:
        raise ValueError(
            'checkpoint does not contain class_to_idx; use a checkpoint saved by '
            'the public-benchmark fine-tuning runner'
        )

    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)
    dataset_test = build_dataset('test', args)
    # TODO: re-enable once IDRiD false-positive (test filenames restart at
    # IDRiD_001, colliding by name with train) is confirmed via content hash.
    # validate_imagefolder_splits(dataset_train, {'val': dataset_val, 'test': dataset_test})
    validate_split_class_maps(checkpoint_class_to_idx, {'train': dataset_train.class_to_idx})
    args.num_labels = len(checkpoint_class_to_idx)
    print(f'Checkpoint class mapping: {checkpoint_class_to_idx}')

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

    classifier_state_dict = checkpoint.get('classifier_state_dict', None)
    if classifier_state_dict is None:
        print("Cannot find the weights for classifier (decoder). Please refer to our fine-tuning instruction!!")
        print("The classifier would utlize the random weights!!")
    else:
        msg = linear_classifier.load_state_dict(classifier_state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    model.eval()
    linear_classifier.eval()
    test_stats, preds, targets, _ = validate_network(
        test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)

    output = np.vstack(preds)
    target = np.vstack(targets)
    test_stats.update(compute_classification_metrics(output, target, args.num_labels))

    print(f"AUC: {test_stats['auc']}, AUPR: {test_stats['aupr']}")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'best.npy'), output)
    np.save(os.path.join(args.output_dir, 'target.npy'), target)
    scalar_stats = {k: float(v) for k, v in test_stats.items() if k != 'epoch'}
    with open(os.path.join(args.output_dir, 'test_stats.json'), 'w') as f:
        json.dump(scalar_stats, f, indent=2)
    print(json.dumps(scalar_stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluating a VisionFM public-benchmark checkpoint on the test split')
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
        help='Checkpoint saved by finetune_visionfm_publicbench_fundus.py.')
    parser.add_argument('--checkpoint_key', default='visionfm_state_dict', type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument('--local_rank', default=0, type=int, help='Please ignore and do not set this argument.')
    parser.add_argument('--data_path', type=str, required=True,
        help='Root folder for one public benchmark dataset, containing train/val/test/Class_x subfolders.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--modality', default='Fundus', type=str)
    parser.add_argument('--task', default='PAPILA', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default='.', help='Path to save logs and checkpoints')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
