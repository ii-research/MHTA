import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.model import ModelWithEMDLoss
from src.dataset import TemporalDurationDataset, TemporalDurationDataset_question, ContrastiveTemporalDataset_question
from src.dataset import ContrastiveTemporalDataset
from src.dataset import TemporalQADataset
from src.dataset import LabelLengthBucketBatchSampler
from src.dataset import GenerativeKeywordDataset
from src.train import train_model
from src.utils import get_duration_token_ids, get_token_ids_from_vocab
from peft import LoraConfig
from src.utils import print_config
from src.utils import setup_full_logging
from src.utils import create_output_dir
from src.utils import custom_collate
from src.utils import triplet_collate
from src.utils import ord_collate


import os
from src.prompt_selector import prompt_selector, prompt_selector_with_question


with open('configs/config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

dynamic_output_dir = create_output_dir(cfg['training']['output_dir'], cfg)
log_dir = os.path.join(dynamic_output_dir, "logs")
setup_full_logging(log_dir)
print_config(cfg)

tokenizer = AutoTokenizer.from_pretrained(cfg['base_model_name'])
if "Hunyuan" in cfg['base_model_name']:
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model_name'],trust_remote_code = True)
else:
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model_name'])



tokenizer.pad_token = tokenizer.eos_token

token_ids_dict = get_token_ids_from_vocab(cfg['vocab_to_keyword'], tokenizer)

print("token ids:", token_ids_dict)

train_ord_dataset = TemporalDurationDataset(cfg["dataset"]["train_ord"], tokenizer, prompt_selector,tiny_data=cfg['debug']['tiny_data'])

train_qa_dataset = TemporalQADataset(cfg["dataset"]["train_qa"], tokenizer, use_few_shot=cfg['training']['qa_use_few_shot'], num_shots=cfg['training']['qa_num_shot'], tiny_data=cfg['debug']['tiny_data'],prompt_template=cfg["evaluation"]["prompt_template"])

train_gen_dataset = GenerativeKeywordDataset(cfg["dataset"]["train_gen"], tokenizer, tiny_data=cfg['debug']['tiny_data'])


train_dataset = TemporalDurationDataset(cfg["dataset"]["train"], tokenizer, prompt_selector,tiny_data=cfg['debug']['tiny_data'])
if cfg['training']['train_generated_question']:
    train_dataset= TemporalDurationDataset_question(cfg["dataset"]["train_with_question"], tokenizer, prompt_selector_with_question, tiny_data=cfg['debug']['tiny_data'])
    # train_dataset= TemporalDurationDataset_question(cfg["dataset"]["train_with_question"], tokenizer, prompt_selector, tiny_data=cfg['debug']['tiny_data'])


#tiny 设为了true
dev_dataset = TemporalDurationDataset(cfg["dataset"]["dev"], tokenizer, prompt_selector,tiny_data=cfg['debug']['tiny_data'])

train_contra_dataset = ContrastiveTemporalDataset(cfg["dataset"]["train_contra"], tokenizer, prompt_selector, tiny_data=cfg['debug']['tiny_data'])
if cfg['training']['train_contra_generated_question']:
    train_contra_dataset= ContrastiveTemporalDataset_question(cfg["dataset"]["train_contra_questions"], tokenizer, prompt_selector_with_question, tiny_data=cfg['debug']['tiny_data'])



train_batch_sampler = LabelLengthBucketBatchSampler(
    dataset=train_dataset,
    batch_size=cfg['training']['batch_size'],       # 设置你希望的训练 batch size     # True = 丢掉小 batch；False = 保留
    shuffle=True         # 每轮打乱样本顺序
)

dev_batch_sampler = LabelLengthBucketBatchSampler(
    dataset=dev_dataset,
    batch_size=cfg['training']['batch_size'],       # 设置你希望的训练 batch size     # True = 丢掉小 batch；False = 保留
    shuffle=True         # 每轮打乱样本顺序
)

train_contra_dataset_batch_sampler = LabelLengthBucketBatchSampler(
    dataset=train_contra_dataset,
    batch_size=cfg['training']['batch_size'],       # 设置你希望的训练 batch size     # True = 丢掉小 batch；False = 保留
    shuffle=True         # 每轮打乱样本顺序
)

# train_dataloader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
# dev_dataloader = DataLoader(dev_dataset, batch_size=cfg['training']['eval_batch_size'], shuffle=True)
# train_contra_dataloader = DataLoader(train_contra_dataset, batch_size=cfg['training']['eval_batch_size'], shuffle=True)

train_qa_dataloader = DataLoader(train_qa_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
train_gen_dataloader = DataLoader(train_gen_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)


train_ord_dataloader = DataLoader(train_ord_dataset, batch_size=cfg['training']['eval_batch_size'], collate_fn=ord_collate, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=custom_collate)
dev_dataloader = DataLoader(dev_dataset, batch_sampler=dev_batch_sampler, collate_fn=custom_collate)
train_contra_dataloader = DataLoader(train_contra_dataset, batch_sampler=train_contra_dataset_batch_sampler, collate_fn=triplet_collate)

model = ModelWithEMDLoss(
    base_model_name=cfg['base_model_name'],
    token_ids_dict=token_ids_dict,
    lora_config=LoraConfig(**cfg['lora_config']),
    tokenizer=tokenizer,
    cfg=cfg
)

train_model(
    model=model,
    tokenizer=tokenizer,
    output_dir=dynamic_output_dir,
    config=cfg,
    train_dataloader=train_dataloader,
    eval_dataloader=dev_dataloader,
    train_contra_dataloader=train_contra_dataloader,
    train_ord_dataloader=train_ord_dataloader,
    train_qa_dataloader=train_qa_dataloader,
    train_gen_dataloader=train_gen_dataloader
)
