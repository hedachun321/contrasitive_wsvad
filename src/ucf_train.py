import torch
from scipy import ndimage
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option


def select_topk_embeddings(scores, features_, k):
    _, idx_DESC = scores.sort(descending=True, dim=1)
    idx_topk = idx_DESC[:, :k]
    idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, features_.shape[2]])
    selected_embeddings = torch.gather(features_, 1, idx_topk)
    return selected_embeddings

def easy_snippets_mining(actionness, fertures_, args):
    actionness = actionness.squeeze()
    select_idx = torch.ones_like(actionness).cuda()
    # select_idx = self.dropout(select_idx)

    actionness_drop = actionness * select_idx

    actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
    actionness_rev_drop = actionness_rev * select_idx

    easy_act_mu = select_topk_embeddings(actionness_drop, fertures_, args.k_easy)

    easy_bkg_mu = select_topk_embeddings(actionness_rev_drop, fertures_, args.k_easy)

    k=max(1, int(fertures_.shape[-2] // args.k_easy))
    return easy_act_mu, easy_bkg_mu

def hard_snippets_mining(actionness, fertures_, args):
    actionness = actionness.squeeze()
    aness_np = actionness.cpu().detach().numpy()
    aness_median = np.median(aness_np, 1, keepdims=True)
    aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

    erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,args.M))).astype(aness_np.dtype)
    erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,args.m))).astype(aness_np.dtype)
    idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
    aness_region_inner = actionness * idx_region_inner
    hard_act_mu = select_topk_embeddings(aness_region_inner, fertures_, k=args.k_hard)

    dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,args.m))).astype(aness_np.dtype)
    dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,args.M))).astype(aness_np.dtype)
    idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
    aness_region_outer = actionness * idx_region_outer
    hard_bkg_mu = select_topk_embeddings(aness_region_outer, fertures_, k=args.k_hard)

    return hard_act_mu, hard_bkg_mu


def Euclidean_distance_single(x, y):
    # 直接计算两个张量之间的欧氏距离
    dist = torch.cdist(x, y)

    return torch.mean(dist)


def KL_divergence_single(x, y):
    # 添加小常数避免除零
    x = x + 1e-5
    y = y + 1e-5

    # 计算 KL 散度的不同项
    term1 = 0.5 * torch.einsum('bd,bd,bd->b', [(y - x), 1 / y, (y - x)])
    term2 = 0.5 * (torch.log(y).sum(-1) - torch.log(x).sum(-1))
    term3 = 0.5 * ((x / y).sum(-1))

    dist = term1 + term2 + term3 - 0.5 * x.shape[1]

    return torch.mean(1 / (dist + 1))


def Bhattacharyya_distance_single(x, y):
    # 添加小常数避免除零
    x = x + 1e-5
    y = y + 1e-5

    # 计算 Bhattacharyya 距离的不同项
    term1 = 0.125 * torch.einsum('bd,bd,bd->b', [(x - y), 2 / (x + y), (x - y)])
    term2 = 0.5 * (torch.log((x + y) / 2).sum(-1) - (torch.log(x).sum(-1) + torch.log(y).sum(-1)))

    dist = term1 + term2

    return torch.mean(1 / (dist + 1))


def Mahalanobis_distance_single(x, y):
    # 假设 x 和 y 是直接输入的单个张量，而不是成对的均值和协方差
    cov_inv = 1 / (x + y + 1e-5)  # 可以根据需要调整 1e-5 以防止除零
    dist = torch.einsum('bd,bd,bd->b', [(x - y), cov_inv, (x - y)])

    return torch.mean(1 / (dist + 1))


def Intra_ProbabilsticContrastive(hard_query, easy_pos, easy_neg, args):


    if args.metric == 'Mahala':
        pos_distance = Mahalanobis_distance_single(hard_query, easy_pos)
        neg_distance = Mahalanobis_distance_single(hard_query, easy_neg)

    elif args.metric == 'KL_div':
        pos_distance = 0.5 * (KL_divergence_single(hard_query, easy_pos) + KL_divergence_single(easy_pos, hard_query))
        neg_distance = 0.5 * (KL_divergence_single(hard_query, easy_neg) + KL_divergence_single(easy_neg, hard_query))

    elif args.metric == 'Bhatta':
        pos_distance = Bhattacharyya_distance_single(hard_query, easy_pos)
        neg_distance = Bhattacharyya_distance_single(hard_query, easy_neg)

    elif args.metric == 'Euclidean':
        pos_distance = Euclidean_distance_single(hard_query, easy_pos)
        neg_distance = Euclidean_distance_single(hard_query, easy_neg)

    if args.loss_type == 'frobenius':
        loss = torch.norm(1 - pos_distance) + torch.norm(neg_distance)
        return loss

    elif args.loss_type == 'neg_log':
        loss = -1 * (torch.log(pos_distance) + torch.log(1 - neg_distance))
        return loss.mean()


def CLASM3(logits, feature_, device, args):
    # 超参

    easy_act, easy_bkg = easy_snippets_mining(logits, feature_, args)
    hard_act, hard_bkg = hard_snippets_mining(logits, feature_, args)

    action_prob_contra_loss = args.alpha5 * Intra_ProbabilsticContrastive(hard_act, easy_act, easy_bkg, args)
    background_prob_contra_loss = args.alpha6 * Intra_ProbabilsticContrastive(hard_bkg, easy_bkg, easy_act, args)


    return action_prob_contra_loss + background_prob_contra_loss

def CLAS4(logits, labels, lengths, device, args):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])  # 将标签转换为目标形式
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    # 前一半为正常视频，后一半为异常视频
    batch_size = logits.shape[0]
    half_batch_size = batch_size // 2

    normal_scores = torch.zeros(half_batch_size).to(device)  # 存储正常视频的均分
    abnormal_scores = torch.zeros(half_batch_size).to(device)  # 存储异常视频的均分
    topk_abnormal_scores = torch.zeros(half_batch_size).to(device)  # 存储异常视频 top-k 片段的均分

    for i in range(batch_size):
        valid_length = lengths[i]
        k = int(valid_length / 16 + 1)

        # 取出有效的 logits 并计算均值
        if i < half_batch_size:
            # 处理正常视频
            normal_scores[i] = torch.mean(logits[i, 0:valid_length])
        else:
            # 处理异常视频
            abnormal_scores[i - half_batch_size] = torch.mean(logits[i, 0:valid_length])
            tmp, _ = torch.topk(logits[i, 0:valid_length], k=k, largest=True)
            topk_abnormal_scores[i - half_batch_size] = torch.mean(tmp)

    # 构建三元对比损失
    # 1. 正常视频的均分 < 异常视频的均分
    normal_vs_abnormal_loss = F.relu(normal_scores - abnormal_scores + args.alpha1).mean()

    # 2. 异常视频的均分 < top-k 异常片段的均分
    abnormal_vs_topk_loss = F.relu(abnormal_scores - topk_abnormal_scores + args.alpha2).mean()

    # 总损失 = 两个对比损失
    total_loss = normal_vs_abnormal_loss + abnormal_vs_topk_loss

    return total_loss





def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLASM_(logits, labels, lengths, device):
    instance_logits1 = torch.zeros(0).to(device)
    instance_logits2 = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    # 假设logits的形状为(总行数, 列数)
    total_rows = logits.shape[0]
    half_rows = total_rows // 2
    # 将logits按第一个维度拆分成两半
    logits_first_half = logits[:half_rows, :]
    logits_second_half = logits[half_rows:, :]
    labels1 = labels[:half_rows, :]
    labels2 = labels[half_rows:, :]

    for i in range(logits_first_half.shape[0]):
        tmp, _ = torch.topk(logits_first_half[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits1 = torch.cat([instance_logits1, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss1 = -torch.mean(torch.sum(labels1 * F.log_softmax(instance_logits1, dim=1), dim=1), dim=0)

    for i in range(logits_second_half.shape[0]):
        logit = logits_second_half[i, 0:lengths[i]]
        window =int(lengths[i] / 16 + 1)
        instance = torch.zeros(1, 0).to(device)
        for j in range(logit.size(1)):
            column_tensor = logit[:, j]
            tmp = rolling_sum_max(column_tensor, window)
            tmp = tmp / window
            tmp = tmp.view(1, -1)  # 将tmp变形成维度为1的张量
            instance = torch.cat([instance, tmp], dim=1)
        instance_logits2 = torch.cat([instance_logits2, instance], dim=0)
    milloss2 = -torch.mean(torch.sum(labels2 * F.log_softmax(instance_logits2, dim=1), dim=1), dim=0)

    milloss = (milloss1 + milloss2) / 2
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

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

def CLAS2DMIL(logits, labels, lengths, device):
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    # 初始化损失值
    clsloss = torch.tensor(0.0).to(device)
    for i in range(logits.shape[0]):
        # 对每个样本进行处理
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        # 计算每个样本的实例损失
        instance_loss = F.binary_cross_entropy(tmp, labels[i].expand_as(tmp))
        # 累加到总损失上
        clsloss += instance_loss
    # 计算平均损失
    clsloss /= logits.shape[0]
    return clsloss

def rolling_sum_max(tensor, window_size):
    unfolded = tensor.unfold(0, window_size, 1)
    windows_sum = unfolded.sum(dim=1)
    max_sum = torch.max(windows_sum, dim=0).values
    min_sum = torch.min(windows_sum, dim=0).values
    return max_sum, min_sum

def CLAS222(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    instance_logits_n = torch.zeros(0).to(device)
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
        tmp, min_n = rolling_sum_max(logits_second_half[i, 0:lengths[i]], k)
        tmp = tmp / k
        min_n = min_n / k
        tmp = tmp.view(1)
        min_n = min_n.view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
        instance_logits_n = torch.cat([instance_logits_n, min_n], dim=0)
    labels2 = torch.ones_like(instance_logits)  # 假设所有样本的目标相似度都为1
    labels3 = torch.zeros_like(instance_logits_n)  # 假设所有样本的目标相似度都为1
    clsloss2 = F.binary_cross_entropy(instance_logits, labels2)
    clsloss3 = F.binary_cross_entropy(instance_logits_n, labels3)
    clsloss = (clsloss2 + clsloss3) / 2
    return (clsloss + clsloss1) / 2

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0
    BEST_AUC =0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
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
            #loss1
            loss1 = CLAS222(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            # #loss2
            # loss2 = CLASM_(logits2, text_labels, feat_lengths, device)
            # loss_total2 += loss2.item()
            # #loss3
            # loss3 = torch.zeros(1).to(device)
            # text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            # for j in range(1, text_features.shape[0]):
            #     text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
            #     loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            # loss3 = loss3 / 13 * 1e-1
            nmloss = normal_smooth(logits1, text_labels, feat_lengths, device)
            # loss4 = CENTROPY(logits1, logits2, feat_lengths, device)
            # loss_total4 += loss4.item()

            loss2 = CLAS4(logits1, text_labels, feat_lengths, device, args)

            loss = loss1 + nmloss * 10 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1))
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                if BEST_AUC < AUC:
                    BEST_AUC = AUC
                print("BEST_AUC:", BEST_AUC)
                
        scheduler.step()
        
        torch.save(model.state_dict(), '../model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    return ap_best

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = ucf_option.parser.parse_args()
    # 网格搜索超参数
    param_grid = {
        'alpha1': [0.1, 0.3, 0.5, 0.7, 0.9],
        'alpha2': [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    import itertools

    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))

    best_score = float('-inf')
    best_params = None

    for combination in combinations:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        setup_seed(args.seed)

        label_map = dict({
            'Normal': 'normal',
            'Ordinary': 'ordinary',  # 88.10
            'Common': 'common',  # 87.96

            'Abuse': 'abuse',
            # 'Trauma': 'trauma', # 88.16

            'Arrest': 'arrest',
            'Justice': 'justice',  # 88.02
            'Handcuffs': 'handcuffs',  # 87.92

            'Arson': 'arson',
            'Fire': 'fire',  # 88.13
            'Destruction': 'destruction',  # 87.99

            'Assault': 'assault',
            'Injury': 'injury',  # 88.11

            'Burglary': 'burglary',
            'Theft': 'theft',  # 88.11
            'Intrusion': 'intrusion',  # 88.05

            'Explosion': 'explosion',
            # 'Disaster': 'disaster',#88.22
            'Debris': 'debris',  # 88.08

            'Fighting': 'fighting',
            'Warfare': 'warfare',  # 88.07

            'RoadAccidents': 'roadAccidents',
            'Vehicle Damage': 'vehicle damage',  # 87.73
            'Traffic Congestion': 'traffic congestion',  # 87.95

            # what other categories are Robbery visually similar to?
            'Robbery': 'robbery',
            'Shooting': 'shooting',
            'Shoplifting': 'shoplifting',
            'Stealing': 'stealing',
            'Vandalism': 'vandalism',
            'Violence': 'violence',  # 88.11
            'Conflict': 'conflict',  # 88.05
            'Victimization': 'victimization',  # 87.95
            'MentalHealth': 'mentalHealth',  # 87.99
            'PowerDynamics': 'powerDynamics',  # 87.73
            'Recovery': 'recovery',  # 87.70
            'Healing': 'healing'  # 87.70
        })
        normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
        normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        params = dict(zip(keys, combination))

        # 更新 args 中的参数
        args.alpha1 = params['alpha1']
        args.alpha2 = params['alpha2']

        # 创建模型
        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                        args.visual_head, args.visual_layers, args.attn_window,
                        args.prompt_prefix, args.prompt_postfix, device)


        # 训练模型并获取评分
        ap_best = train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)

        if ap_best > best_score:
            best_score = ap_best
            best_params = params

        print("Currenet alpha5:", args.alpha5)
        print("Currenet alpha6:", args.alpha6)
        print("currenet Best Auc:", best_score)

    print("Best Score:", best_score)
    print("Best Params:", best_params)