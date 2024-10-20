import os

import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, save_dir):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            logits1 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])

            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)

            prob1_np = prob1.cpu().numpy()
            prob1_repeated = np.repeat(prob1_np, 16)
            t_np = gt[i]

            # 生成并保存当前样本的异常分数图
            plt.figure(figsize=(12, 6))
            plt.plot(prob1_repeated, color='red', linewidth=3)
            # plt.rcParams.update({'font.size': 18})  # 设置全局字体大小为14

            # 用透明红色背景高亮显示 gtsegments
            for k, segment in enumerate(gtsegments[i]):
                start, end = map(int, segment)
                if gtlabels[i][k] == 'A':
                    continue
                plt.gca().add_patch(Rectangle((start, 0), end - start, 1.02, color='dodgerblue', alpha=0.8))

            plt.ylim(0, 1.02)  # 设置 y 轴范围从 0 到 1
            plt.yticks(np.arange(0, 1.2, 0.2))  # 设置 y 轴刻度间隔为 0.2
            plt.xlim(0, len(prob1_repeated) - 1)
            # 或者可以分别设置 x 和 y 轴刻度的字体大小
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            # plt.title(f'Sample {i} Anomaly Score')
            plt.xlabel('Frame', fontsize=18)
            plt.ylabel('Anomaly Score', fontsize=18)

            plt.legend().remove()
            plt.savefig(os.path.join(save_dir, f'anomaly_score_sample_{i}.svg'), format='svg', bbox_inches='tight')
            plt.close()

    ap1 = ap1.cpu().numpy()
    ap1 = ap1.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)

    return ROC1, AP1, 0  # , averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict(
        {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
         'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                    args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)
    save_dir = '/media/wang/Disk/1_hwc/VadCLIP_88.49_窗口loss/visual_xd_result3_84_20'
    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, save_dir)
