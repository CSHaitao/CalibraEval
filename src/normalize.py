
import json
from math import exp

path = 'data.json'
out_path = 'logit.json'
out_file = open(out_path,'w',encoding='utf-8')
file = open(path,'r')

num = 0 
for line in file.readlines():
    a_dict = json.loads(line)
    num = num + 1
    
    
    model_pair = a_dict['model_pair']

    prompt_1 = a_dict['prompt_1']
    prompt_2 = a_dict['prompt_2']
    prompt_3 = a_dict['prompt_3']
    
    
    save_dict = {}
    save_dict['qid'] = a_dict['qid']
    save_dict['model_pair'] = model_pair


    A_exp = 0
    B_exp = 0
    try:
        for token,logit in zip(prompt_1['tokens'],prompt_1['logits']):
            if token == 'A':
                A_exp = exp(logit)
            if token == 'B':
                B_exp = exp(logit)

        prompt_1_logit = {'A': A_exp/(A_exp+B_exp) , 'B': B_exp/(A_exp+B_exp)}
    except:
        continue
    
    save_dict['prompt_1_logit'] = prompt_1_logit       
    
    A_exp = 0
    B_exp = 0
    try:
        for token,logit in zip(prompt_2['tokens'],prompt_2['logits']):
            if token == 'A':
                A_exp = exp(logit)
            if token == 'B':
                B_exp = exp(logit)

        prompt_2_logit = {'A': A_exp/(A_exp+B_exp) , 'B': B_exp/(A_exp+B_exp)}
    except:
        continue
    save_dict['prompt_2_logit'] = prompt_2_logit       
     
    A_exp = 0
    B_exp = 0
    try:
        for token,logit in zip(prompt_3['tokens'],prompt_3['logits']):
            if token == 'A':
                A_exp = exp(logit)
            if token == 'B':
                B_exp = exp(logit)

        prompt_3_logit = {'A': A_exp/(A_exp+B_exp) , 'B': B_exp/(A_exp+B_exp)}
    except:
        continue
    save_dict['prompt_3_logit'] = prompt_3_logit    
    out_file.write(json.dumps(save_dict) + '\n')
