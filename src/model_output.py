
import json
import openai
import httpx
from openai import APIConnectionError, OpenAI, RateLimitError
from api import Auto_API
# from api import Auto_API
from loguru import logger
from tqdm import tqdm


openai_client = OpenAI(
    base_url=, 
    api_key=,
    http_client=httpx.Client(
        base_url=,
        follow_redirects=True,
    ),
    )

out_path = 'output.json'
out_file = open(out_path,'a',encoding='utf-8')



def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": messages,
            },
        ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-logprobs",
                messages=messages,
                temperature = 0,
                logprobs=True,
                top_logprobs=5,
            )
            # print(response)
            # print(response.choices[0].message.content)
            # print(response.choices[0].logprobs)
            return response
        except (APIConnectionError, RateLimitError):
            logger.warning(
                f"API call failed on attempt {attempt + 1}, retrying..."
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


path = 'data.json'
file = open(path,'r')

for line in tqdm(file.readlines()):
    
    
    
    a_dict = json.loads(line)
    
    aaa = a_dict['question_id']


    
    prompt_1 = a_dict['prompt_1']
    prompt_2 = a_dict['prompt_2']
    prompt_3 = a_dict['prompt_3']
    
    response_1 = attempt_api_call(openai_client,EVALUATION_MODEL_NAME,prompt_1)
    if response_1 == None:
        continue
    response_content = response_1.choices[0].message.content
 
    logprobs_1 = response_1.choices[0].logprobs.content[0].top_logprobs
    tokens_1 = []
    logits_1 = []
    
    for aa in logprobs_1:
        tokens_1.append(aa.token)
        logits_1.append(aa.logprob)
   


    response_2 = attempt_api_call(openai_client,EVALUATION_MODEL_NAME,prompt_2)
    if response_2 == None:
            continue
    response_content = response_2.choices[0].message.content
 
    logprobs_2 = response_2.choices[0].logprobs.content[0].top_logprobs
    tokens_2 = []
    logits_2 = []
    
    for aa in logprobs_2:
        tokens_2.append(aa.token)
        logits_2.append(aa.logprob)
        
        
    response_3 = attempt_api_call(openai_client,EVALUATION_MODEL_NAME,prompt_3)
    if response_3 == None:
        continue
    response_content = response_3.choices[0].message.content
  
    logprobs_3 = response_3.choices[0].logprobs.content[0].top_logprobs
    tokens_3 = []
    logits_3 = []
    
    for aa in logprobs_3:
        tokens_3.append(aa.token)
        logits_3.append(aa.logprob)
  
    save_dict = {}
    save_dict['qid'] = a_dict['question_id']
    
 
    save_dict['model_pair'] = a_dict['model_pair']
    save_dict['prompt_1'] = {'tokens':tokens_1, 'logits': logits_1}
    save_dict['prompt_2'] = {'tokens':tokens_2, 'logits': logits_2}
    save_dict['prompt_3'] = {'tokens':tokens_3, 'logits': logits_3}
    out_file.write(json.dumps(save_dict) + '\n')

