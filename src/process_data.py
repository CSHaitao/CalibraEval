
import json
import random
prompt_templete = """###Task: Given a question and two answers. Determine which one better answers the question. You only need to output "A" or "B" directly to indicate which answer is better.

###Query: 
{question}

###A Answer: 
{A}

###B Answer:
{B}

###Output:
"""

prompt_templete_2 = """###Task: Given a question and two answers. Determine which one better answers the question. You only need to output "A" or "B" directly to indicate which answer is better.

###Query: 
{question}

###B Answer: 
{B}

###A Answer:
{A}

###Output:
"""

out_path = '/reward_process.json'
out_file = open(out_path,'w',encoding='utf-8')
path = 'reward.json'

file = open(path,'r')


for line in file.readlines():
    a_dict = json.loads(line)
    
    query = a_dict['prompt']
    chosen = a_dict['chosen']
    chosen_model = a_dict['chosen_model']
    rejected = a_dict['rejected']
    rejected_model = a_dict['rejected_model']
    subset = a_dict['subset']
    id  = a_dict['id']
    
    random_number = random.uniform(0, 1)
    if random_number >= 0.5:
        label = -1
        model_pair = [chosen_model,rejected_model]
        A = chosen
        B = rejected
    else:
        label = 1
        model_pair = [rejected_model,chosen_model]
        B = chosen
        A = rejected      
    
    
    prompt_1 = prompt_templete.format(
                question=query, A=A, B = B
            )
    prompt_2 = prompt_templete_2.format(
                question=query, A=A, B = B
            )
    prompt_3 = prompt_templete.format(
                question=query, A=B, B = A
            )
    
    
    save_dict = {}
    save_dict['id'] = id
    save_dict['label'] = label
    save_dict['model_pair'] = model_pair
    save_dict['model_a'] = A
    save_dict['model_b'] = B
    save_dict['prompt_1'] = prompt_1
    save_dict['prompt_2'] = prompt_2
    save_dict['prompt_3'] = prompt_3
    out_file.write(json.dumps(save_dict) + '\n')
