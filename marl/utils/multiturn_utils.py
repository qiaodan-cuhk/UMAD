
from glob import glob
import openai
import json
import numpy as np
import time
import random
import os
import transformers
import torch
from tqdm import tqdm
import argparse

def generate_answer_summary(answer_context, model = "mistral", tokenizer = None, hf_model = None, device = None):
    if model not in ["mistral", "phi3", "llama3"]:
        try:
            completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0125",
                    seed=0,
                    messages=answer_context,
                    n=1)
        except:
            print("retrying due to an error......")
            time.sleep(20)
            return generate_answer_summary(answer_context)
    else:
        hf_model = hf_model.to(device)
        input_text = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = hf_model.generate(input_ids, max_length=len(input_ids[0]) + 2048, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, top_p = 0.9, temperature = 1)
        generated_ids = output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
        cpu_device = torch.device("cpu")
        hf_model = hf_model.to(cpu_device)
    return completion


def generate_answer(answer_context, i, model, models, device = None, tokenizer = None):
    if model not in ["mistral", "phi3", "llama3"]:
        try:
            completion = openai.ChatCompletion.create(
                    model=models[i%3],
                    messages=answer_context,
                    seed=i,
                    n=1)
        except:
            print("retrying due to an error......")
            time.sleep(20)
            return generate_answer(answer_context, i, model, models)
    else:
        hf_model = models[i%3]
        hf_model = hf_model.to(device)
        input_text = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = hf_model.generate(input_ids, max_length=len(input_ids[0]) + 2048, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, top_p = 0.9, temperature = 1)
        generated_ids = output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
        cpu_device = torch.device("cpu")
        hf_model = hf_model.to(cpu_device)
    return completion

def load_hf_model(model_path):
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True)
    except:
        raise OSError(f"{model_path} does not exist or there was an error during finetuning...")
    return model

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def summarize_message(agent_contexts, hf_model = None, tokenizer = None, device = None):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent and explain the reasoning in each solution."
    agent_context = [{"role": "user", "content": prefix_string}]
    completion = generate_answer_summary(agent_context, hf_model = hf_model, tokenizer = tokenizer, device = device)
    content = completion["choices"][0]["message"]["content"]

    return content