"""
File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import re

from openai import OpenAI
import openai
import time

def temp_sleep(seconds=0.1):
    time.sleep(seconds)


def ChatGPT_single_request(prompt):
    temp_sleep()

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    temp_sleep()

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion["choices"][0]["message"]["content"]

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"


def ChatGPT_request(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    # temp_sleep()
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion["choices"][0]["message"]["content"]

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"



def GPT_request(prompt, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    temp_sleep()
    try:
        response = openai.Completion.create(
            model=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"], )
        return response.choices[0].text
    except:
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classifiy) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final promopt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r", encoding='utf-8')
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        if i == 'None':
            i = ""
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    # remove seconds from time
    pattern = r"(\d{2}:\d{2}):00"

    # Replace ':00' at the end of times with nothing
    modified_prompt = re.sub(pattern, r"\1", prompt)
    modified_prompt = re.sub(r'\bYou\b', 'you', modified_prompt, flags=re.IGNORECASE)
    modified_prompt = re.sub(r'\bYour\b', 'your', modified_prompt, flags=re.IGNORECASE)
    cleaned_prompt = '\n'.join([line for line in modified_prompt.split('\n') if line.strip()])
    return cleaned_prompt.strip()


def execute_prompt(prompt, llm, objective, history=None, temperature=0.6):
    print(f"==============={objective}=========================")

    # 1. 针对 DeepSeek 的 Prompt 强化（强制它按格式输出，不要废话）
    # 我们在 prompt 后面追加一段强制指令
    enforced_prompt = prompt + "\n\nIMPORTANT: Please answer strictly in the expected format (e.g., 'Key: Value'). Do not output any conversational filler."

    response = None
    while response is None:
        try:
            client = OpenAI(
                base_url="http://10.10.63.35:8000/v1",
                api_key="EMPTY"
            )
            
            messages = []
            if history is None:
                messages = [{"role": "user", "content": enforced_prompt}]
            else:
                messages = history
                # 如果是 history 模式，我们在最后一条消息追加指令
                if messages and messages[-1]['role'] == 'user':
                    messages[-1]['content'] += "\n\nIMPORTANT: Answer strictly in 'Key: Value' format."

            completion = client.chat.completions.create(
                model=llm.model,
                messages=messages,
                temperature=temperature,
            )
            response = completion
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            print('Retrying...')
            time.sleep(2)

    answer = response.choices[0].message.content
    
    # --- [关键修改 1: 打印原始输出，方便你看 DeepSeek 到底说了啥] ---
    print(f"\n--- [DEBUG] Raw Output (First 100 chars) ---\n{answer[:100]}...\n--------------------------------\n")

    # --- [关键修改 2: 强力清洗 <think> 标签] ---
    # 解释：(?:</think>|$) 意思是如果找不到结尾标签，就删到字符串末尾（防止截断导致正则失效）
    answer = re.sub(r'<think>.*?(?:</think>|$)', '', answer, flags=re.DOTALL)
    
    # --- [关键修改 3: 兜底修复] ---
    # 如果清洗后是空的（说明模型只思考没回答），或者格式不对，我们手动“伪造”一个格式让程序不崩溃
    answer = answer.strip()
    if not answer: 
        print("Warning: Model output empty after cleaning, using fallback.")
        return "Role: Unknown" # 这是一个兜底，防止 split 报错
    
    # 如果没有冒号，大概率是模型在罗嗦，我们尝试强行提取或返回原样
    if ":" not in answer and len(answer) > 0:
        # 很多时候 DeepSeek 会直接说 "Student"，我们帮它补上 "Role: "
        if "role" in objective.lower():
             return f"Role: {answer}"
    
    return answer


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


