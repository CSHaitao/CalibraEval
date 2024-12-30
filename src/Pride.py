'''
Author: lihaitao
Date: 2024-09-17 20:08:27
LastEditors: Do not edit
LastEditTime: 2024-10-09 22:35:27
FilePath: /eval_LLM/Code_1005/Pride.py
'''
import os
import json
import math
import argparse
import random
import numpy as np
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing as mp
from sklearn.metrics import classification_report
from itertools import permutations
import pandas as pd
import pingouin as pg
import joblib
import krippendorff
import numpy as np
from sklearn.metrics import classification_report

RATIO_PREFIX_SAMPLES = 1
def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x

def fleiss_kappa(table):
    """
    计算 Fleiss's kappa 多评分者一致性测量.

    参数:
    table : array_like, 2-D
        假定行为主体在行，类别在列

    返回值:
    kappa : float
        Fleiss's kappa 一致性统计量

    备注:
    代码参考自 Wikipedia 页面
    http://en.wikipedia.org/wiki/Fleiss%27_kappa
    """
    table = np.asarray(table, dtype=float)  # 避免整型除法
    n_sub, n_cat = table.shape
    n_total = table.sum()
    n_rater = table.sum(axis=1)
    n_rat = n_rater.max()

    assert n_total == n_sub * n_rat, "总评分数量不匹配"

    p_cat = table.sum(axis=0) / n_total
    p_rat = (table ** 2).sum(axis=1) - n_rat
    p_rat /= (n_rat * (n_rat - 1))
    p_mean = p_rat.mean()

    p_mean_exp = (p_cat ** 2).sum()
    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return kappa

def process_data(file_path,label_dict=None,prior=None):
    k1, k2, k3 = [], [], []
    all_data = []
    all_kappa = []
    all_num = 0
    correct_num_1 = 0
    correct_num_2 = 0
    correct_num_3 = 0
    labels = []
    predictions_1 = []
    predictions_2 = []
    predictions_3 = []
    with open(file_path, 'r') as file:
        for line in file:
            one_kappa = [0, 0]
            a_dict = json.loads(line)
            
            qid = a_dict['qid']
            # if type != 'Reasoning':
            #     continue
            obe_1 = np.array([a_dict['prompt_1_logit']['A'],a_dict['prompt_1_logit']['B']])
            obe_2 = np.array([a_dict['prompt_2_logit']['A'],a_dict['prompt_2_logit']['B']])
            obe_3 = np.array([a_dict['prompt_3_logit']['A'],a_dict['prompt_3_logit']['B']])
            
            
            de_1 = np.log(obe_1 + 1e-10) - np.log(prior + 1e-10)
            de_2 = np.log(obe_2 + 1e-10) - np.log(prior + 1e-10)
            de_3 = np.log(obe_3 + 1e-10) - np.log(prior + 1e-10)
            # predictions.append(np.argmax(de_1))
            # print(softmax(de_1))
            A_1_logit = softmax(de_1)[0]
            A_2_logit = softmax(de_2)[0]
            A_3_logit = softmax(de_3)[0]
            
            all_data.append([A_1_logit, A_2_logit, 1 - A_3_logit])
            
            # 计算kappa
            one_kappa[0] += sum([A_1_logit > 0.5, A_2_logit > 0.5, A_3_logit <= 0.5])
            one_kappa[1] += sum([A_1_logit <= 0.5, A_2_logit <= 0.5, A_3_logit > 0.5])
            all_kappa.append(one_kappa)

            k1.append(int(A_1_logit > 0.5))
            k2.append(int(A_2_logit > 0.5))
            k3.append(int(A_3_logit <= 0.5))
            
            if label_dict:
                qid = a_dict['qid']
                label = label_dict[str(qid)]
                ##计算方差
                if label==0:
                    continue
                if label == -1:
                    labels.append(0)
                if label == 1:
                    labels.append(1)
                if A_1_logit >= 0.5:
                    predictions_1.append(0)
                if A_1_logit < 0.5:
                    predictions_1.append(1)
                if A_2_logit >= 0.5:
                    predictions_2.append(0)
                if A_2_logit < 0.5:
                    predictions_2.append(1)
                if A_3_logit < 0.5:
                    predictions_3.append(0)
                if A_3_logit >= 0.5:
                    predictions_3.append(1)
                
                all_num += 1
                
                if A_1_logit >= 0.5 and label == -1:
                    correct_num_1 += 1
                if A_1_logit < 0.5 and label == 1:
                    correct_num_1 += 1
                if A_2_logit >= 0.5 and label == -1:
                    correct_num_2 += 1
                if A_2_logit < 0.5 and label == 1:
                    correct_num_2 += 1      
                if A_3_logit <= 0.5 and label == -1:
                    correct_num_3 += 1
                if A_3_logit > 0.5 and label == 1:
                    correct_num_3 += 1  
            

            

    # print(len(predictions))
    # print(all_data)
    # final_score = np.mean(np.array(predictions) == np.array(labels)) 
    # print(final_score)
    # print(all_kappa)
    fleiss = fleiss_kappa(all_kappa)
    print(f"Fleiss's Kappa: {fleiss}")
    
    # ratings = np.array(all_data).T
    # alpha = krippendorff.alpha(reliability_data=ratings, level_of_measurement='ratio')

    # print(f"Krippendorff's Alpha: {alpha}")

    df = pd.DataFrame(all_data, columns=['Rater1', 'Rater2', 'Rater3'])
    df_long = df.melt(var_name='Rater', value_name='Rating', ignore_index=False).reset_index()
    df_long.rename(columns={'index': 'Target'}, inplace=True)


    # 计算ICC
    icc = pg.intraclass_corr(data=df_long, targets='Target', raters='Rater', ratings='Rating')
    print(icc)
    # icc1 = icc[icc['Type'] == 'ICC1']
    # print("ICC1",icc1)
    # icc2 = icc[icc['Type'] == 'ICC2']
    # print("ICC2",icc2)
    report = classification_report(labels, predictions_1, output_dict=True)
    recalls = [report[str(e)]['recall'] * 100 for e in range(2)]
    std_1  = np.std(recalls)
    print("std_1",std_1)
    report = classification_report(labels, predictions_2, output_dict=True)
    recalls = [report[str(e)]['recall'] * 100 for e in range(2)]
    std_2  = np.std(recalls)
    print("std_2",std_2)
    report = classification_report(labels, predictions_3, output_dict=True)
    recalls = [report[str(e)]['recall'] * 100 for e in range(2)]
    std_3  = np.std(recalls)
    print("std_3",std_3)
    
    print((std_1+std_2+std_3)/3)
    
    if label_dict:
        print("prompt_1 Accuracy",correct_num_1/all_num)
        print("prompt_2 Accuracy",correct_num_2/all_num)
        print("prompt_3 Accuracy",correct_num_3/all_num)
        print(((correct_num_1/all_num)+(correct_num_2/all_num)+(correct_num_3/all_num))/3)
            









def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x

def debias_fn(observed, permuted_indices=None):
    observed = np.array(observed)
    observed = observed / (observed.sum(axis=1, keepdims=True) + 1e-10)

    if permuted_indices is None:
        if observed.shape[1] == 2:
            permuted_indices = [
                (0, 1,),
                (1, 0,),
            ]
        else:
            permuted_indices = [
                (0, 1, 2, 3, 4),
                (1, 2, 3, 4, 0),
                (2, 3, 4, 0, 1),
                (3, 4, 0, 1, 2),
                (4, 0, 1, 2, 3),
            ]

   
    debiased = gather_probs(observed, permuted_indices)
    debiased = np.mean(debiased, axis=1)
    prior = softmax(np.log(observed + 1e-10).mean(axis=0))

    return observed, debiased, prior

def gather_probs(observed, permuted_indices=None):
    if permuted_indices is None:
        permuted_indices = sorted(permutations(range(observed.shape[1])))
    assert len(permuted_indices) == observed.shape[0]
    gathered_probs = [[] for _ in range(observed.shape[1])]

    for pdx, indices in enumerate(permuted_indices):
        for idx, index in enumerate(indices):
            gathered_probs[index].append(observed[pdx, idx])
    return gathered_probs




def single_process(all_probs,source_rng):
    n_iters = 5
    if RATIO_PREFIX_SAMPLES == 1.:
        n_iters = 1
        
    num_prefix_samples = int(len(all_probs) * RATIO_PREFIX_SAMPLES)

    costs = []
    scores = []
    recall_stds = []
    for iter_idx in range(n_iters):
        predictions = []
        labels = []
        cost = []

        source_rng.shuffle(all_probs)
        prefix_samples = all_probs[:num_prefix_samples]
        postfix_samples = all_probs[num_prefix_samples:]

        all_priors = []
        all_observed = []
        for idx, prefix_sample in enumerate(prefix_samples):
            observed, ideal = prefix_sample
            observed = np.array(observed)
            observed, debiased, prior = debias_fn(observed)
            all_priors.append(prior)
            all_observed.append(observed)
            predictions.append(np.argmax(debiased))
            cost.append(len(observed))
            labels.append(ideal)

        prior = np.mean(all_priors, axis=0)
        # print(prior)
        for postfix_sample in postfix_samples:
            observed, ideal = postfix_sample
            observed = np.array(observed[0])
            debiased = np.log(observed + 1e-10) - np.log(prior + 1e-10)
            predictions.append(np.argmax(debiased))
            cost.append(1)
            labels.append(ideal)

        # print(len(labels))
        # print(len(predictions))
        final_score = np.mean(np.array(predictions) == np.array(labels)) * 100
        # print('final_score',final_score)
        scores.append(final_score)
        costs.append(np.mean(cost))
        
        # print(len(labels))
        # print(predictions)

        report = classification_report(labels, predictions, output_dict=True)
        recalls = [report[str(e)]['recall'] * 100 for e in range(prior.shape[-1])]
        recall_stds.append(np.std(recalls))
    # print(recall_stds)
    # print(float(np.mean(recall_stds)))
    res = {
        'rstd': float(np.mean(recall_stds)),
        'rstd_max': float(np.max(recall_stds)),
        'rstd_min': float(np.min(recall_stds)),
        'rstd_std': float(np.std(recall_stds)),
        'acc': float(np.mean(scores)),
        'acc_max': float(np.max(scores)),
        'acc_min': float(np.min(scores)),
        'acc_std': float(np.std(scores)),
        'cost': float(np.mean(costs)),
    }
    print(res)
    return res,prior



def load_result(path,label_path):
    label_dict = {}
    label_file = open(label_path,'r')
    for line in label_file.readlines():
        a_dict = json.loads(line)
        if a_dict['label'] == -1:
            label = 0
        else:
            label = 1
        label_dict[str(a_dict['id'])] = label
    
    file = open(path,'r')
    all_probs = []
    labels = []
    for line in file.readlines(): 
        a_dict = json.loads(line)
        qid = a_dict['qid']
        label = label_dict[str(qid)]
        labels.append(label)
        prob = []
        prob.append([a_dict['prompt_1_logit']['A'],a_dict['prompt_1_logit']['B']])  
        prob.append([a_dict['prompt_3_logit']['A'],a_dict['prompt_3_logit']['B']]) 
        all_probs.append(prob)
    return list(zip(all_probs, labels))

path = '/home/lht/eval_LLM/Code_1005/data/output/gpt-3.5-turbo-logprobs_7_109_2.json'
label_path  = '/home/lht/eval_LLM/Code_0827/label/reward_label_raw.json'
source_rng = random.Random(path.encode('utf-8'))
all_probs = load_result(path,label_path)

res,prior = single_process(all_probs,source_rng)
print(prior)


label_dict = {}
label_file = open(label_path,'r')
for line in label_file.readlines():
    a_dict = json.loads(line)
    label = a_dict['label']
    label_dict[str(a_dict['id'])] = label



process_data(path,label_dict=label_dict,prior=prior)

