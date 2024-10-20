import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import xd_option


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def normal_smooth(logits, labels, lengths, device):
    normal_smooth_loss = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        logit = logits[i, 0:lengths[i]]
        if labels[i] == 0:
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(logit).unsqueeze(0)))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)

    return normal_smooth_loss

def rolling_sum_max(tensor, window_size):
    unfolded = tensor.unfold(0, window_size, 1)
    windows_sum = unfolded.sum(dim=1)
    max_sum = torch.max(windows_sum, dim=0).values
    return  max_sum

def CLAS222(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    # 初始化损失值
    clsloss1 = torch.tensor(0.0).to(device)

    # 假设logits的形状为(总行数, 列数)
    total_rows = logits.shape[0]
    half_rows = total_rows // 2
    # 将logits按第一个维度拆分成两半
    logits_first_half = logits[:half_rows, :]
    logits_second_half = logits[half_rows:, :]


    for i in range(logits_first_half.shape[0]):
        # 对每个样本进行处理
        tmp, _ = torch.topk(logits_first_half[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        # 计算每个样本的实例损失
        instance_loss = F.binary_cross_entropy(tmp, labels[i].expand_as(tmp))
        # 累加到总损失上
        clsloss1 += instance_loss

    # 计算平均损失
    clsloss1 /= logits_first_half.shape[0]

    for i in range(logits_second_half.shape[0]):
        k = int(lengths[i] / 16 + 1)
        tmp = rolling_sum_max(logits_second_half[i, 0:lengths[i]], k)
        tmp = tmp / k
        tmp = tmp.view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    labels2 = torch.ones_like(instance_logits)  # 假设所有样本的目标相似度都为1
    clsloss = F.binary_cross_entropy(instance_logits, labels2)
    return (clsloss + clsloss1) / 2



def train(model, normal_loader, anomaly_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.000005)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            logits1 = model(visual_features, None, prompt_text, feat_lengths)

            loss1 = CLAS222(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            nmloss = normal_smooth(logits1, text_labels, feat_lengths, device)


            loss = loss1 + nmloss * 20

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 2400 == 0 and step != 0:
                print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1))
                AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels,
                                    device)
                if AP > ap_best:
                    ap_best = AP
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                print("BEST_AUC:", ap_best)

        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    # label_map = dict(
    #     {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
    #      'G': 'explosion'})
    label_map = dict(
        {'A': 'normal',
         'B1': 'fighting',
         'B11': 'boxing',
         'B12': 'wrestling',
         'B2': 'shooting',
         'B21': 'gunfights',
         'B4': 'riot',
         'B41': 'conflict',
         'B42': 'turmoil',
         'B5': 'abuse',
         'B51': 'mistreatment',
         'B52': 'bullying',
         'B6': 'car accident',
         'B61': 'collision',
         'B62': 'crash',#82.22
         'G': 'explosion',
         'G1': 'blast',#82.25
         'G2': 'burst'})

    # train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    normal_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                    args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)