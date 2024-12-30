'''
Author: lihaitao
Date: 2024-06-18 16:15:21
LastEditors: Do not edit
LastEditTime: 2024-10-09 23:17:59
FilePath: /eval_LLM/Code_1005/iso_regression_batch.py
'''
import json
from math import exp
import random
import os
import numpy as np
import json
import numpy as np
import pandas as pd
import pingouin as pg
import joblib
from sklearn.isotonic import IsotonicRegression
os.makedirs('calibration', exist_ok=True)


file = open(f'data.json', 'r')



def balanced_sampling(data, num_samples_per_bin=1000, num_bins=20):
    """
    对0-1之间的值进行均衡取样。

    参数:
    - data: 输入的数据数组，包含0到1之间的值。
    - num_samples_per_bin: 每个区间内要抽取的样本数量。
    - num_bins: 区间的数量。

    返回:
    - sampled_data: 均衡取样后的数据数组。
    """
    # 确定区间边界
    bin_edges = np.linspace(0, 1, num_bins + 1)

    # 使用字典存储每个区间的样本
    bin_dict = {i: [] for i in range(num_bins)}

    # 将数据分配到对应的区间
    for value in data:
        bin_index = np.digitize(value[0], bin_edges) - 1
        if 0 <= bin_index < num_bins:
            bin_dict[bin_index].append(value)

    # 进行均衡抽样
    sampled_data = []
    for bin_index in range(num_bins):
        bin_samples = bin_dict[bin_index]
        if len(bin_samples) > num_samples_per_bin:
            sampled_data.extend(random.sample(bin_samples, num_samples_per_bin))
        else:
            sampled_data.extend(bin_samples)

    return sampled_data





logits = []
samples = []
list_num = [0,0,0,0,0,0,0,0,0,0]
for line in file.readlines():
    a_dict = json.loads(line)

    prompt_1 = a_dict['prompt_1_logit']['A']
    prompt_2 = a_dict['prompt_2_logit']['A']
    prompt_3 = a_dict['prompt_3_logit']['A']
    
 
    samples.append([prompt_1, prompt_2, prompt_3])


## 是否抽样
# samples = balanced_sampling(samples)
for one in samples:
    logits.append(one[0])
    logits.append(one[1])
    logits.append(one[2])
print(len(logits))
print(len(samples))



def iso_regression(yhats, samples_num, lam=0.1, batch_size=32):
    values = list(set(yhats + [-0.1, 1.1]))
    values.sort()
    N = len(values) - 1
    values2idx = {v: idx  for idx, v in enumerate(values)} ##值对应index
    yhatidxs = [values2idx[y] for y in yhats]  ### index的list
    samples = []
    for i in range(int(len(yhats)/3)):
        samples.append([yhatidxs[3*i], yhatidxs[3*i+1],yhatidxs[3*i+2]]) 


    paras = [0. for _ in range(N)]

    def calculate_loss(paras, samples):
        exps = [exp(p) for p in paras]
        sums = [0.]
        for _ in exps:
            sums.append(sums[-1] + _)
        total = sums[-1]
        _loss, _regular = 0., 0.
        for s1, s2, s3 in samples:
            _loss += (sums[s1] / total + sums[s3] / total - 1.) ** 2  ###loss
            _loss += (sums[s1] / total - sums[s2] / total) ** 2
            _regular += (sums[s1] / total - sums[s3] / total) ** 2
        return _loss/len(samples), _regular/len(samples)


    def calculate_loss_and_gradient(paras, batch_samples):
        exps = [exp(p) for p in paras]
        sums = [0.]
        for _ in exps:
            sums.append(sums[-1] + _)
        total = sums[-1]
        _loss, _regular = 0., 0.
        grads = [0. for _ in range(N)]

        for s1, s2, s3 in batch_samples:
            grad1 = 2 * (sums[s1] / total + sums[s3] / total - 1.)
            grad2 = 2 * lam * (sums[s1] / total - sums[s2] / total)
            grad3 = 2 * lam * (sums[s1] / total - sums[s3] / total)

            for i in range(s1):
                grads[i] += (grad1 + grad2 + grad3) * (-exps[s1 - 1]) / (total ** 2)
            for i in range(s1, N):
                grads[i] += (grad1 + grad2 + grad3) * exps[s1 - 1] * (1. - sums[s1] / total) / total

            for i in range(s2):
                grads[i] += (grad1 - grad2 - grad3) * (-exps[s1 - 1]) / (total ** 2)
            for i in range(s2, N):
                grads[i] += (grad1 - grad2 - grad3) * exps[s1 - 1] * (1. - sums[s1] / total) / total

            for i in range(s3):
                grads[i] += (grad1 - grad2 - grad3) * (-exps[s1 - 1]) / (total ** 2)
            for i in range(s3, N):
                grads[i] += (grad1 - grad2 - grad3) * exps[s1 - 1] * (1. - sums[s1] / total) / total

            _loss += (sums[s1] / total + sums[s3] / total - 1.) ** 2
            _loss += (sums[s1] / total - sums[s2] / total) ** 2
            _regular += (sums[s1] / total - sums[s3] / total) ** 2

        return _loss, _regular, grads

    lr = 10
    epoch = 1

 
    

    for iter_num in range(epoch):
        loss, reg = calculate_loss(paras, samples)
        print(f"iter@{iter_num}: {loss}, {reg}, {loss-lam*reg}")
        
        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start:batch_start + batch_size]
            loss, reg, grads = calculate_loss_and_gradient(paras, batch_samples)
            # print(grads[0])
            for i in range(N):
                paras[i] += lr * grads[i]/batch_size
        _s = sum(paras) / N
        paras = [p-_s for p in paras]
        
        

        exps = [exp(p) for p in paras]
        sums = [0.]
        for _ in exps:
            sums.append(sums[-1] + _)
        total = sums[-1]

        funcs = dict()
        for i in range(1, N):
            funcs[float(values[i])] = sums[i] / total
            

        
        new_sample = []
        for one in samples_num:
            new_sample.append([funcs[one[0]],funcs[one[1]],1-funcs[one[2]]])
        df = pd.DataFrame(new_sample, columns=['Rater1', 'Rater2', 'Rater3'])
        df_long = df.melt(var_name='Rater', value_name='Rating', ignore_index=False).reset_index()
        df_long.rename(columns={'index': 'Target'}, inplace=True)
        icc = pg.intraclass_corr(data=df_long, targets='Target', raters='Rater', ratings='Rating')
        icc1 = icc[icc['Type'] == 'ICC2k']
        print("校准后",icc1)





    exps = [exp(p) for p in paras]
    sums = [0.]
    for _ in exps:
        sums.append(sums[-1] + _)
    total = sums[-1]

    funcs = dict()
    for i in range(1, N):
        funcs[float(values[i])] = sums[i] / total

    
    x_train = []
    for one in funcs.keys():
        x_train.append(float(one))

    y_train = []
    for one in funcs.values():
        y_train.append(float(one))

    X_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train)

    print(X_train.shape)
    print(y_train.shape)
    # quantile_regressor = QuantileRegressor(quantile=0.5, alpha=0)  # 中位数回归（保序回归的一种）
    # quantile_regressor.fit(X_train, y_train)
    # joblib.dump(quantile_regressor, 'isotonic_regression_model.pkl')
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(X_train, y_train)

    # 保存模型
    joblib.dump(iso_reg, 'model.pkl')
    json.dump(funcs, open(f'model.json', 'w'))

    return

iso_regression(logits, samples, lam=0.05, batch_size = 64)