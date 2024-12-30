
import json
import numpy as np
import pandas as pd
import pingouin as pg
import joblib
import krippendorff
import numpy as np
from sklearn.metrics import classification_report
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





def process_data(file_path, calibration_dict=None,label_dict=None,type_dict=None):
    k1, k2, k3 = [], [], []
    all_data = []
    all_kappa = []
    all_num = 0
    correct_num_1 = 0
    correct_num_2 = 0
    correct_num_3 = 0
    correct_num_4 = 0
    labels = []
    predictions_1 = []
    predictions_2 = []
    predictions_3 = []
    aggre_num = 0
    acc_score = 0
    new_pre = 0.5
    with open(file_path, 'r') as file:
        for line in file:
            one_kappa = [0, 0]
            a_dict = json.loads(line)
            
            qid = a_dict['qid']
          
            
            
            A_1_logit = a_dict['prompt_1_logit']['A']
            A_2_logit = a_dict['prompt_2_logit']['A']
            A_3_logit = a_dict['prompt_3_logit']['A']


            if calibration_dict:
                A_1_logit = calibration_dict.predict(np.array(A_1_logit).reshape(-1, 1))[0]
                A_2_logit = calibration_dict.predict(np.array(A_2_logit).reshape(-1, 1))[0]
                A_3_logit = calibration_dict.predict(np.array(A_3_logit).reshape(-1, 1))[0]
                
 
               
            
            
            A_logit = (A_1_logit +A_2_logit+1-A_3_logit )/3
            all_data.append([A_1_logit, A_2_logit, 1 - A_3_logit])
            
            new_pre = 0.5
            if sum([A_1_logit > 0.5, A_2_logit > 0.5, A_3_logit <= 0.5]) == 3 :
                aggre_num += 1
                new_pre = 0
            if sum([A_1_logit <= 0.5, A_2_logit <= 0.5, A_3_logit > 0.5]) == 3:
                aggre_num += 1
                new_pre = 1
            
                
      
            one_kappa[0] += sum([A_1_logit > 0.5, A_2_logit > 0.5, A_3_logit <= 0.5])
            one_kappa[1] += sum([A_1_logit <= 0.5, A_2_logit <= 0.5, A_3_logit > 0.5])
            all_kappa.append(one_kappa)

            k1.append(int(A_1_logit > 0.5))
            k2.append(int(A_2_logit > 0.5))
            k3.append(int(A_3_logit <= 0.5))
            
            if label_dict:
                qid = a_dict['qid']
                label = label_dict[str(qid)]
           
                if label == 0:
                    continue
                if label == -1:
                    labels.append(0)
                    if new_pre == 0:
                        acc_score += 1
                if label == 1:
                    labels.append(1)
                    if new_pre == 1:
                        acc_score += 1
                if new_pre == 0.5:
                    acc_score += 0.5

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
              
                 
                if A_logit>=0.5 and label == -1:
                    correct_num_4 += 1
                if A_logit<0.5 and label == 1:
                    correct_num_4 += 1 
            

            

    fleiss = fleiss_kappa(all_kappa)
    print(f"Fleiss's Kappa: {fleiss}")
    
   

    df = pd.DataFrame(all_data, columns=['Rater1', 'Rater2', 'Rater3'])
    df_long = df.melt(var_name='Rater', value_name='Rating', ignore_index=False).reset_index()
    df_long.rename(columns={'index': 'Target'}, inplace=True)


    # 计算ICC
    icc = pg.intraclass_corr(data=df_long, targets='Target', raters='Rater', ratings='Rating')
    print(icc)
    ## RSTD
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
        print(" Accuracy",correct_num_4/all_num)
            

# 处理未校准的数据
file_path = 'data.json'
label_path = 'label.json'


label_file = open(label_path,'r')
label_dict = {}
type_dict = {}
for line in label_file.readlines():
    a_dict = json.loads(line)
    label_dict[str(a_dict['id'])] = a_dict['label']
print("未校准数据:")
process_data(file_path,label_dict=label_dict,type_dict=type_dict)

# 处理校准后的数据


iso_reg = joblib.load('/home/lht/eval_LLM/Code_1005/model/reward_model_qwen_3shot.pkl')

print("校准后数据:")
process_data(file_path, calibration_dict=iso_reg,label_dict=label_dict,type_dict=type_dict)   
    




