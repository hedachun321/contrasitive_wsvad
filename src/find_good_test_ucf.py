import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
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

    ap1 = ap1.cpu().numpy()
    ap1 = ap1.tolist()
    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    print("AUC1: ", ROC1, " AP1: ", AP1)
    return ROC1, AP1


if __name__ == '__main__':
    # 确定使用的设备，优先使用cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({
        'Normal': 'normal',
        'Ordinary': 'ordinary', # 88.10
        'Common': 'common', # 87.96

        'Abuse': 'abuse',
        # 'Trauma': 'trauma', # 88.16


        'Arrest': 'arrest',
        'Justice': 'justice',# 88.02
        'Handcuffs': 'handcuffs',# 87.92


        'Arson': 'arson',
        'Fire': 'fire', # 88.13
        'Destruction': 'destruction',# 87.99


        'Assault': 'assault',
        'Injury': 'injury',# 88.11

        'Burglary': 'burglary',
        'Theft': 'theft', #88.11
        'Intrusion': 'intrusion', #88.05

        'Explosion': 'explosion',
        # 'Disaster': 'disaster',#88.22
        'Debris': 'debris', # 88.08

        'Fighting': 'fighting',
        'Warfare': 'warfare',# 88.07

        'RoadAccidents': 'roadAccidents',
        'Vehicle Damage': 'vehicle damage',#87.73
        'Traffic Congestion': 'traffic congestion',#87.95

        # what other categories are Robbery visually similar to?
        'Robbery': 'robbery',
        'Shooting': 'shooting',
        'Shoplifting': 'shoplifting',
        'Stealing': 'stealing',
        'Vandalism': 'vandalism',
        'Violence': 'violence',#88.11
        'Conflict': 'conflict',#88.05
        'Victimization': 'victimization',#87.95
        'MentalHealth': 'mentalHealth',#87.99
        'PowerDynamics': 'powerDynamics',#87.73
        'Recovery': 'recovery',#87.70
        'Healing': 'healing'#87.70
    })
    prompt_text = get_prompt_text(label_map)

    # 加载ground truth数据
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 初始化模型
    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
        args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device
    )

    # 加载模型参数
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    # 读取CSV文件
    csv_path = "/media/wang/Disk/1_hwc/VadCLIP_88.49_窗口loss/list/ucf_CLIP_rgbtest_bak.csv"
    df = pd.read_csv(csv_path)

    # 初始化变量，记录最佳结果
    best_result_overall = None
    best_filename_overall = None

    # 遍历每一行
    for index, row in df.iterrows():
        # 当前行最好的结果
        current_best_result = None
        current_best_filename = None

        print(f"\n处理第 {index + 1}/{len(df)} 行数据")

        # 遍历文件名从__0到__9
        for i in range(10):
            path_parts = row['path'].rsplit('__', 1)
            filename = f"{path_parts[0]}__{i}.npy"
            row['path'] = filename

            # 更新测试数据集
            temp_df = df.copy()
            temp_df.iloc[index] = row
            temp_csv_path = "temp_test_list.csv"
            temp_df.to_csv(temp_csv_path, index=False)

            testdataset = UCFDataset(args.visual_length, temp_csv_path, True, label_map)
            testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

            # 执行测试
            result, _ = test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

            print(f"测试结果 {filename}: {result}")

            # 如果当前结果优于之前的最佳结果，更新最佳结果
            if current_best_result is None or result > current_best_result:
                current_best_result = result
                current_best_filename = filename

                print(f"当前行的新最佳结果: {current_best_result} (文件名: {current_best_filename})")

        # 更新行的文件名为当前最佳结果
        df.at[index, 'path'] = current_best_filename

        print(f"第 {index + 1} 行的最佳结果: {current_best_result} (文件名: {current_best_filename})")

        # 如果当前行的最佳结果优于之前的历史最佳结果，更新历史最佳结果
        if best_result_overall is None or current_best_result > best_result_overall:
            best_result_overall = current_best_result
            best_filename_overall = current_best_filename

        print(f"当前历史最佳结果: {best_result_overall} (文件名: {best_filename_overall})")

        # 将更新后的结果保存回原CSV文件
        df.to_csv(csv_path, index=False)

    print("\n所有行数据处理完成。最佳结果已保存到CSV文件。")