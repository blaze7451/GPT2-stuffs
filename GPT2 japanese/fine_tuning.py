import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import glob

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

novel_path = glob.glob("C:\\Python\\Pytorch\\Transformer related\\Fine tuning GPT 2\\GPT2 japanese\\*.txt")[0]

