import pprint
import sys
import os
import datetime
import torch
import tqdm


def triplet_collate(batch):
    return {
        "anchor_input_ids": torch.stack([x["anchor_input_ids"] for x in batch]),
        "anchor_attention_mask": torch.stack([x["anchor_attention_mask"] for x in batch]),
        "anchor_labels": torch.stack([x["anchor_label"] for x in batch]),
        "anchor_indices": [x["anchor_indices"] for x in batch],

        "positive_input_ids": torch.stack([x["positive_input_ids"] for x in batch]),
        "positive_attention_mask": torch.stack([x["positive_attention_mask"] for x in batch]),
        "positive_labels": torch.stack([x["positive_label"] for x in batch]),
        "positive_indices": [x["positive_indices"] for x in batch],

        "negative_input_ids": torch.stack([x["negative_input_ids"] for x in batch]),
        "negative_attention_mask": torch.stack([x["negative_attention_mask"] for x in batch]),
        "negative_labels": torch.stack([x["negative_label"] for x in batch]),
        "negative_indices": [x["negative_indices"] for x in batch],
    }

def custom_collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "indices": [x["indices"] for x in batch],    # list of list[float]
    }

def ord_collate(batch):
    return {
        "ord_input_ids": torch.stack([x["ord_input_ids"] for x in batch]),
        "ord_attention_mask": torch.stack([x["ord_attention_mask"] for x in batch]),
        "ord_labels": torch.stack([x["ord_labels"] for x in batch]),
        "ord_indices": [x["ord_indices"] for x in batch],    # list of list[float]
    }

def remove_padding(indices, values, pad_threshold=120):
    filtered_indices = []
    filtered_values = []
    for idx, val in zip(indices, values):
        if idx < pad_threshold:
            filtered_indices.append(idx)
            filtered_values.append(val)
    return filtered_indices, filtered_values
def get_duration_token_ids(duration_units, tokenizer):
    token_ids = []
    for unit in duration_units:
        token = tokenizer.tokenize(unit)
        tok_id = tokenizer.convert_tokens_to_ids(token)
        token_ids.append(tok_id[0])
    return token_ids

def get_token_ids_from_vocab(vocab_to_keyword, tokenizer):
    token_ids = {}
    for idx, keyword in vocab_to_keyword.items():
        token = tokenizer.tokenize(keyword)
        tok_id = tokenizer.convert_tokens_to_ids(token)
        token_ids[idx] = tok_id[0]  # 只取第一个 token 的 id
    return token_ids

def print_config(config):
    print("\n========== Loaded Configuration ==========")
    pprint.pprint(config, indent=2)
    print("==========================================\n")


# def setup_full_logging(log_dir):
# #     os.makedirs(log_dir, exist_ok=True)
# #     log_path = os.path.join(log_dir, "full_run.log")
# #     log_file = open(log_path, "w", buffering=1)
# #
# #     class Logger(object):
# #         def __init__(self):
# #             self.terminal = sys.stdout
# #             self.log = log_file
# #
# #         def write(self, message):
# #             self.terminal.write(message)
# #             self.log.write(message)
# #
# #         def flush(self):
# #             self.terminal.flush()
# #             self.log.flush()
# #
# #
# #     sys.stdout = Logger()
# #     sys.stderr = Logger()

# def setup_full_logging(log_dir):
#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, "full_run.log")
#     log_file = open(log_path, "w", buffering=1)
#
#     class Logger(object):
#         def __init__(self):
#             self.terminal = sys.__stdout__  # use original stdout
#             self.log = log_file
#
#         def write(self, message):
#             self.terminal.write(message)
#             self.log.write(message)
#
#         def flush(self):
#             self.terminal.flush()
#             self.log.flush()
#
#     logger = Logger()
#     sys.stdout = logger
#     sys.stderr = logger
#
#     # Ensure tqdm also respects this redirection
#     import tqdm
#     tqdm.utils._term_move_up = lambda *a, **kw: None  # optional: avoid cursor movement codes
#     tqdm.tqdm.__init__ = (lambda old_init: lambda self, *args, **kwargs: old_init(self, *args, file=sys.stdout, **kwargs))(tqdm.tqdm.__init__)
def setup_full_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "full_run.log")
    log_file = open(log_path, "w", buffering=1)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.__stdout__  # original stdout
            self.log = log_file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    logger = Logger()
    sys.stdout = logger
    sys.stderr = logger

    # ✅ 安全版本的 patch：仅当 file 没有被传入时才加入 sys.stdout
    old_init = tqdm.tqdm.__init__

    def safe_tqdm_init(self, *args, **kwargs):
        if 'file' not in kwargs:
            kwargs['file'] = sys.stdout
        return old_init(self, *args, **kwargs)

    tqdm.tqdm.__init__ = safe_tqdm_init

def create_output_dir(base_output_dir, cfg):

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg['base_model_name'].split('/')[-1]
    # lr = cfg['training']['learning_rate']
    epochs = cfg['training']['num_epochs']
    train_gen = cfg['training']["train_gen"]

    use_cost = cfg['training']['use_cost']
    mode = cfg['training']['mode']
    # 创建自定义的目录名，包含时间和关键参数
    if train_gen:
        output_dir_name = f"{model_name}_{mode}_cost{str(use_cost)}_train_gen_epochs{epochs}_{now}"
    else:
        output_dir_name = f"{model_name}_{mode}_cost{str(use_cost)}_epochs{epochs}_{now}"
    output_dir = os.path.join(base_output_dir, output_dir_name)

    # 创建文件夹
    os.makedirs(output_dir, exist_ok=True)
    return output_dir