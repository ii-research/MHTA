from torch.utils.data import Dataset
from src.utils import remove_padding
from torch.utils.data import Sampler
import random
from collections import defaultdict
from src.prompt_selector import build_temporal_qa_prompt
import torch
from torch.utils.data import Dataset
import json

class LabelLengthBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.buckets = defaultdict(list)
        for idx in range(len(dataset)):
            label_len = dataset.get_label_length(idx)
            self.buckets[label_len].append(idx)

        self.batch_indices = self._build_batches()

    def _build_batches(self):
        all_batches = []
        for bucket in self.buckets.values():
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)

class TemporalDurationDataset(Dataset):
    def __init__(self, path, tokenizer, prompt_selector, tiny_data=False):
        """
        path: path to the dataset file
        tokenizer: tokenizer object
        prompt_selector: a function that takes (sentence, question, label_vals) and returns a formatted prompt string
        """
        self.tokenizer = tokenizer
        self.samples = []

        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for idx, line in enumerate(f):
                if line.strip() == "":
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                type = parts[0].strip()
                question = parts[1].strip()
                # print(parts[2].strip().split())
                try:
                    indices = list(map(int, parts[2].strip().split()))
                    label_values = list(map(float, parts[3].strip().split()))
                except:
                    continue

                indices,label_values = remove_padding(indices,label_values)

                prompt = prompt_selector(type, question, indices)
                tokenized = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

                if type == "ORD":
                    self.samples.append({
                        "ord_input_ids": tokenized["input_ids"].squeeze(0),
                        "ord_attention_mask": tokenized["attention_mask"].squeeze(0),
                        "ord_indices": indices,
                        "ord_labels": torch.tensor(label_values, dtype=torch.float)
                        # "labels": label_values

                    })
                else:
                    self.samples.append({
                        "input_ids": tokenized["input_ids"].squeeze(0),
                        "attention_mask": tokenized["attention_mask"].squeeze(0),
                        "indices": indices,
                        "labels": torch.tensor(label_values, dtype=torch.float)
                        # "labels": label_values

                    })

                if tiny_data and len(self.samples) >= 5:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_label_length(self, idx):
        return len(self.samples[idx]["labels"])


class TemporalDurationDataset_question(Dataset):
    def __init__(self, path, tokenizer, prompt_selector, tiny_data=False):
        """
        path: path to the dataset file
        tokenizer: tokenizer object
        prompt_selector: a function that takes (sentence, question, label_vals) and returns a formatted prompt string
        """
        self.tokenizer = tokenizer
        self.samples = []

        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for idx, line in enumerate(f):
                if line.strip() == "":
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                type = parts[0].strip()
                context = parts[1].strip()
                question = parts[4].strip()
                # print(parts[2].strip().split())
                try:
                    indices = list(map(int, parts[2].strip().split()))
                    label_values = list(map(float, parts[3].strip().split()))
                except:
                    continue

                indices,label_values = remove_padding(indices,label_values)


                prompt = prompt_selector(type, context, question, indices)
                tokenized = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

                if type == "ORD":
                    self.samples.append({
                        "ord_input_ids": tokenized["input_ids"].squeeze(0),
                        "ord_attention_mask": tokenized["attention_mask"].squeeze(0),
                        "ord_indices": indices,
                        "ord_labels": torch.tensor(label_values, dtype=torch.float)
                        # "labels": label_values

                    })
                else:
                    self.samples.append({
                        "input_ids": tokenized["input_ids"].squeeze(0),
                        "attention_mask": tokenized["attention_mask"].squeeze(0),
                        "indices": indices,
                        "labels": torch.tensor(label_values, dtype=torch.float)
                        # "labels": label_values

                    })

                if tiny_data and len(self.samples) >= 5:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_label_length(self, idx):
        return len(self.samples[idx]["labels"])



class TemporalQADataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512, tiny_data=False,
                 use_few_shot=False, num_shots=0, shot_selection="random", prompt_template=1):
        """
        Dataset for temporal QA classification.

        Args:
            path (str): Path to TSV file.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Max input token length.
            tiny_data (bool): If True, only load 100 examples.
            use_few_shot (bool): Whether to include few-shot examples.
            num_shots (int): Number of few-shot examples to prepend.
            shot_selection (str): "random" or "first".
        """
        self.tokenizer = tokenizer
        self.samples = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split("\t")
                if len(parts) != 5:
                    continue

                context, question, candidate, label_text, dimension = parts
                label = 1 if label_text.strip().lower() == "yes" else 0

                prompt = build_temporal_qa_prompt(
                    passage=context,
                    question=question,
                    answer=candidate,
                    use_few_shot=use_few_shot,
                    num_shots=num_shots,
                    shot_selection=shot_selection,
                    prompt_template=prompt_template
                )

                tokenized = self.tokenizer(prompt, truncation=True, padding='max_length',
                                           max_length=max_length, return_tensors='pt')

                self.samples.append({
                    "qa_input_ids": tokenized["input_ids"].squeeze(0),
                    "qa_attention_mask": tokenized["attention_mask"].squeeze(0),
                    "qa_label": torch.tensor(label, dtype=torch.long)
                })

                if tiny_data and len(self.samples) >= 5:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ContrastiveTemporalDataset(Dataset):
    def __init__(self, path, tokenizer, prompt_selector, tiny_data=False):
        self.samples = []
        self.tokenizer = tokenizer
        self.prompt_template = prompt_selector

        with open(path, "r", encoding="utf-8") as f:
            headers = f.readline().strip().split("\t")  
            col_idx = {h: i for i, h in enumerate(headers)}  

            for line in f:
                parts = line.strip().split("\t")
                try:
                    anchor_values = list(map(float, parts[col_idx["anchor_values"]].split()))
                    positive_values = list(map(float, parts[col_idx["positive_values"]].split()))
                    negative_values = list(map(float, parts[col_idx["negative_values"]].split()))

                    anchor_indices = list(map(int, parts[col_idx["anchor_indices"]].split()))
                    positive_indices = list(map(int, parts[col_idx["positive_indices"]].split()))
                    negative_indices = list(map(int, parts[col_idx["negative_indices"]].split()))

                    anchor_indices, anchor_values = remove_padding(anchor_indices, anchor_values)
                    positive_indices, positive_values = remove_padding(positive_indices, positive_values)
                    negative_indices, negative_values = remove_padding(negative_indices, negative_values)

                except Exception as e:
                    print(f"[ERROR] Line skipped due to: {e}")
                    print(f"--> Raw line: {line.strip()}")
                    continue

                label_type = parts[col_idx["label_type"]]

                def encode(label_type, sentence, indices):
                    prompt = self.prompt_template(label_type, sentence, indices)
                    return self.tokenizer(prompt, truncation=True, padding="max_length", max_length=512,
                                          return_tensors="pt")

                anchor = encode(label_type, parts[col_idx["anchor_text"]], anchor_indices)
                positive = encode(label_type, parts[col_idx["positive_text"]], positive_indices)
                negative = encode(label_type, parts[col_idx["negative_text"]], negative_indices)

                self.samples.append({
                    "anchor_input_ids": anchor["input_ids"].squeeze(0),
                    "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
                    "anchor_label": torch.tensor(anchor_values, dtype=torch.float),
                    "anchor_indices": anchor_indices,

                    "positive_input_ids": positive["input_ids"].squeeze(0),
                    "positive_attention_mask": positive["attention_mask"].squeeze(0),
                    "positive_label": torch.tensor(positive_values, dtype=torch.float),
                    "positive_indices": positive_indices,

                    "negative_input_ids": negative["input_ids"].squeeze(0),
                    "negative_attention_mask": negative["attention_mask"].squeeze(0),
                    "negative_label": torch.tensor(negative_values, dtype=torch.float),
                    "negative_indices": negative_indices,
                })

                if tiny_data and len(self.samples) >= 5:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_label_length(self, idx):
        return len(self.samples[idx]["anchor_label"])



class ContrastiveTemporalDataset_question(Dataset):
    def __init__(self, path, tokenizer, prompt_selector, tiny_data=False):
        self.samples = []
        self.tokenizer = tokenizer
        self.prompt_template = prompt_selector

        with open(path, "r", encoding="utf-8") as f:
            headers = f.readline().strip().split("\t")  # 读取列名
            col_idx = {h: i for i, h in enumerate(headers)}  # 建立列名到索引的映射

            for line in f:
                parts = line.strip().split("\t")
                try:
                    anchor_values = list(map(float, parts[col_idx["anchor_values"]].split()))
                    positive_values = list(map(float, parts[col_idx["positive_values"]].split()))
                    negative_values = list(map(float, parts[col_idx["negative_values"]].split()))

                    anchor_indices = list(map(int, parts[col_idx["anchor_indices"]].split()))
                    positive_indices = list(map(int, parts[col_idx["positive_indices"]].split()))
                    negative_indices = list(map(int, parts[col_idx["negative_indices"]].split()))

                    anchor_indices, anchor_values = remove_padding(anchor_indices, anchor_values)
                    positive_indices, positive_values = remove_padding(positive_indices, positive_values)
                    negative_indices, negative_values = remove_padding(negative_indices, negative_values)

                except Exception as e:
                    print(f"[ERROR] Line skipped due to: {e}")
                    print(f"--> Raw line: {line.strip()}")
                    continue

                label_type = parts[col_idx["label_type"]]

                def encode(label_type, sentence, question, indices):
                    prompt = self.prompt_template(label_type, sentence, question, indices)
                    return self.tokenizer(prompt, truncation=True, padding="max_length", max_length=512,
                                          return_tensors="pt")

                anchor = encode(label_type, parts[col_idx["anchor_text"]], parts[col_idx["anchor_question"]], anchor_indices)
                positive = encode(label_type, parts[col_idx["positive_text"]], parts[col_idx["positive_question"]],positive_indices)
                negative = encode(label_type, parts[col_idx["negative_text"]], parts[col_idx["negative_question"]],negative_indices)

                self.samples.append({
                    "anchor_input_ids": anchor["input_ids"].squeeze(0),
                    "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
                    "anchor_label": torch.tensor(anchor_values, dtype=torch.float),
                    "anchor_indices": anchor_indices,

                    "positive_input_ids": positive["input_ids"].squeeze(0),
                    "positive_attention_mask": positive["attention_mask"].squeeze(0),
                    "positive_label": torch.tensor(positive_values, dtype=torch.float),
                    "positive_indices": positive_indices,

                    "negative_input_ids": negative["input_ids"].squeeze(0),
                    "negative_attention_mask": negative["attention_mask"].squeeze(0),
                    "negative_label": torch.tensor(negative_values, dtype=torch.float),
                    "negative_indices": negative_indices,
                })

                if tiny_data and len(self.samples) >= 5:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_label_length(self, idx):
        return len(self.samples[idx]["anchor_label"])
