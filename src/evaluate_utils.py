import os
import torch
from tqdm import tqdm
import pandas as pd

import json, os, torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score
import os, json, uuid, re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import os, json, uuid, re
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize, sent_tokenize
import os, json, uuid
from tqdm import tqdm
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# from evaluate import load

from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import numpy as np
import uuid
import json
import os


def evaluate_model(model, eval_dataloader, mode="duration"):
    device = model.model.device
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating [{mode}]"):
            # for k in batch:
            #     batch[k] = batch[k].to(device)
            batch["input_ids"].to(device)
            batch["attention_mask"].to(device)
            batch["labels"].to(device)
            loss = model.compute_loss(batch, mode=mode)
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_count += batch["input_ids"].size(0)

    model.train()
    return total_loss / total_count

def evaluate_on_mctaco(model, tokenizer, device, epoch, output_dir, config):
    import pandas as pd
    import torch
    import os
    from tqdm import tqdm
    from collections import defaultdict


    mctaco_file = config["dataset"]["test"]
    batch_size = config["evaluation"]["batch_size"]
    save_predictions = config["evaluation"].get("save_predictions", True)
    prompt_style = config["evaluation"].get("prompt_style", "standard")
    prompt_template = config["evaluation"].get("prompt_template", 1)
    num_shots = config["evaluation"]["num_shots"]
    dataset = pd.read_csv(mctaco_file, sep='\t',
                          names=["passage", "question", "answer", "label", "reasoning_type"])

    if config['debug']['tiny_data']:
        dataset = dataset[:50]
    if prompt_template == 2:
        pos_label = 'yes'
        neg_label = 'no'
    elif prompt_template == 3:
        pos_label = 'correct'
        neg_label = 'incorrect'
    else:
        pos_label = 'true'
        neg_label = 'false'

    cot_suffix = "\nThink step by step." if prompt_style == "cot" else ""

    fewshot_examples = {
        1: [
            "Is the following candidate answer to the question true or false according to the passage?\nPassage: the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\nQuestion: What happened before Islam was the majority religion?\nCandidate answer: christianity was the majority religion\nThe answer is: true",
            "Is the following candidate answer to the question true or false according to the passage?\nPassage: It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\nQuestion: How long was the storm?\nCandidate answer: 6 years\nThe answer is: false",
            "Is the following candidate answer to the question true or false according to the passage?\nPassage: His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\nQuestion: What time did the battle end?\nCandidate answer: 7:00 PM\nThe answer is: true",
            "Is the following candidate answer to the question true or false according to the passage?\nPassage: In 1930, the poet Muhammad Iqbal proposed a separate Muslim homeland in the northwest of India.\nQuestion: How long did Muhammad Iqbal consider his proposal?\nCandidate answer: 0.56 seconds\nThe answer is: false",
            "Is the following candidate answer to the question true or false according to the passage?\nPassage: He then imprisons the royal family in his prison.\nQuestion: What happened after word spread of the royal family being imprisoned?\nCandidate answer: he and his family doing odd jobs\nThe answer is: false"
        ],
        2: [
            "Based on the information presented in the passage \"the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\", can the candidate answer \"christianity was the majority religion\" answer the question \"What happened before Islam was the majority religion?\"? The answer is: yes",
            "Based on the information presented in the passage \"It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\", can the candidate answer \"6 years\" answer the question \"How long was the storm?\"? The answer is: no",
            "Based on the information presented in the passage \"His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\", can the candidate answer \"7:00 PM\" answer the question \"What time did the battle end?\"? The answer is: yes"
        ],
        3: [
            "According to the passage \"the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\", is the candidate answer \"christianity was the majority religion\" correct to the question \"What happened before Islam was the majority religion?\"? The answer is correct",
            "According to the passage \"It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\", is the candidate answer \"6 years\" correct to the question \"How long was the storm?\"? The answer is incorrect",
            "According to the passage \"His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\", is the candidate answer \"7:00 PM\" correct to the question \"What time did the battle end?\"? The answer is correct"
        ]
    }

    fs_prefix = "\n\n".join(fewshot_examples.get(prompt_template, [])[:num_shots]) + ("\n\n" if num_shots > 0 else "")

    def make_prompt(passage, question, answer, label):
        if prompt_template == 2:
            return fs_prefix + f"Based on the information presented in the passage \"{passage}\", can the candidate answer \"{answer}\" answer the question \"{question}\"?{cot_suffix} The answer is: {label}"
        elif prompt_template == 3:
            return fs_prefix + f"According to the passage \"{passage}\", is the candidate answer \"{answer}\" correct to the question \"{question}\"?{cot_suffix} The answer is {label}"
        else:
            return fs_prefix + f"Is the following candidate answer to the question true or false according to the passage?\nPassage: {passage}\nQuestion: {question}\nCandidate answer: {answer}{cot_suffix}\nThe answer is: {label}"

    prompts_true = [make_prompt(p, q, a, pos_label) for p, q, a in zip(dataset["passage"], dataset["question"], dataset["answer"])]
    prompts_false = [make_prompt(p, q, a, neg_label) for p, q, a in zip(dataset["passage"], dataset["question"], dataset["answer"])]

    gold_labels = dataset["label"].str.strip().str.lower().tolist()
    gold_labels_bin = [1 if g == "yes" else 0 for g in gold_labels]

    predictions = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts_true), batch_size), desc="Evaluating on MC-TACO"):
            pt_batch = prompts_true[i:i + batch_size]
            pf_batch = prompts_false[i:i + batch_size]

            inputs_true = tokenizer(pt_batch, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs_false = tokenizer(pf_batch, return_tensors="pt", padding=True, truncation=True).to(device)

            logits_true = model.model(**inputs_true).logits
            logits_false = model.model(**inputs_false).logits

            last_true_tokens = inputs_true.input_ids[:, -1]
            last_false_tokens = inputs_false.input_ids[:, -1]

            logprob_true = torch.log_softmax(logits_true[:, -2, :], dim=-1).gather(1, last_true_tokens.unsqueeze(1)).squeeze(1)
            logprob_false = torch.log_softmax(logits_false[:, -2, :], dim=-1).gather(1, last_false_tokens.unsqueeze(1)).squeeze(1)

            predictions += (logprob_true >= logprob_false).long().tolist()

    if save_predictions:
        save_path = os.path.join(output_dir, f"mctaco_eval_epoch{epoch + 1}_shot{num_shots}.txt")
        with open(save_path, "w") as f:
            for r in predictions:
                f.write(("yes" if r == 1 else "no") + "\n")

    # Metrics
    key_map = defaultdict(list)
    pred_map = defaultdict(list)
    gold_map = defaultdict(list)

    for i, (p, q) in enumerate(zip(dataset["passage"], dataset["question"])):
        key = f"{p} ||| {q}"
        key_map[key].append(predictions[i] == gold_labels_bin[i])
        pred_map[key].append(predictions[i])
        gold_map[key].append(gold_labels_bin[i])

    strict_acc = sum(all(corrects) for corrects in key_map.values()) / len(key_map)

    avg_f1 = 0.0
    micro_tp = micro_fp = micro_fn = 0
    for key in key_map:
        pred = pred_map[key]
        gold = gold_map[key]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred, gold))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred, gold))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred, gold))
        precision = tp / (tp + fp) if tp + fp > 0 else 1.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        avg_f1 += f1
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    avg_f1 /= len(key_map)
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp > 0 else 1.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn > 0 else 1.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0.0
    acc = sum(p == g for p, g in zip(predictions, gold_labels_bin)) / len(gold_labels_bin)

    # Per-category accuracy
    category_correct = {}
    category_total = {}
    for i, cat in enumerate(dataset["reasoning_type"]):
        cat = cat.strip()
        is_correct = predictions[i] == gold_labels_bin[i]
        category_correct[cat] = category_correct.get(cat, 0) + int(is_correct)
        category_total[cat] = category_total.get(cat, 0) + 1

    category_accuracy = {
        cat: category_correct[cat] / category_total[cat]
        for cat in category_total
    }

    print(f"[MC-TACO] Strict Accuracy: {strict_acc:.4f}")
    print(f"[MC-TACO] Average F1:      {avg_f1:.4f}")
    print(f"[MC-TACO] Micro F1:        {micro_f1:.4f}")
    print(f"[MC-TACO] Normal Accuracy: {acc:.4f}")
    print("[MC-TACO] Accuracy by Reasoning Type:")
    for cat, acc_ in category_accuracy.items():
        print(f" - {cat:20s}: {acc_:.4f}")

    model.train()
    return {
        "strict_acc": strict_acc,
        "avg_f1": avg_f1,
        "micro_f1": micro_f1,
        "acc": acc,
        "category_accuracy": category_accuracy
    }



def evaluate_on_td(model, tokenizer, device, epoch, output_dir, config):

    dataset_path = config["dataset"]["test_td"]
    batch_size = config["evaluation"].get("batch_size", 16)
    save_predictions = config["evaluation"].get("save_predictions", True)
    num_shots = config["evaluation"]["td_num_shots"]
    td_prompt_style=config["evaluation"]["td_prompt_style"]


    # Load dataset
    data = []

    few_shot_examples = [
        "Dialogue:\nA:Hello.Is this room service ?\nB: Yes.May I help you ?\nA: This is room 1425.We asked for the room service an hour ago .\nB: We're very sorry to cause you a lot of inconvenience .\nA: What's the matter ?\nB: We're rather busy right now.It will take <MASK> .\nA: Is it really going to take that long ? Will you rush the order ?\nB: I'm afraid it would take 15 minutes at most .\nA: Ah , well , we have no choice .\n\nIs it appropriate to fill in the <MASK> with \"another 20 minutes\" in this dialogue?\nThe answer is: yes",

        "Dialogue:\nA:A guy in my office got the flu the other day . Today I seem to have come down with it , too .\nB: Very likely . You have a slight fever . Do you have a headache , too ?\nA: Yes . I wonder if you could do something to help me recover soon , because I'll be on a business trip in two days .\nB: Well , you have to let your flu run its course . You must stop working and stay in bed to get plenty of rest . Usually it will take <MASK> to make a full recovery .\nA: But I'm going on a business trip in two days !\nB: Maybe you'll have to cancel it or postpone it . If you go out while you are sick , it won't help you recover . You may even pass your disease on to others .\n\nIs it appropriate to fill in the <MASK> with \"5 to 7 days\" in this dialogue?\nThe answer is: yes",

        "Dialogue:\nA:What schools have you attended ?\nB: I finished Young Primary School in 1998 , and entered Xi ' an Middle School that same September . I graduated from there in <MASK> , and that September I entered Wuhan University , where I'm studying now .\nA: How do you think the education you have received will contribute to your work in this company ?\nB: I think I have a good understanding of fundamentals in the areas your company deals with , and I can go on from here to build up the specific skills and knowledge I need to do my job well .\n\nIs it appropriate to fill in the <MASK> with \"1998\" in this dialogue?\nThe answer is: no",

        "Dialogue:\nA:This is today ' s schedule . At 8: 30AM , conference with the department managers . At 9 o ' clock , live for the workshop where you ' ll award prizes to the staff for preventatives .\nB: That ' s great . What are the prizes ?\nA: 3000 RIB as bonus for each person .\nB: To encourage the staff increases .\nA: Ok . Next thing is laying the corner-stone for the new plant at <MASK> . At 12 AM , back here for lunch .\nB: What about the afternoon ?\nA: At 2 PM , give a presentation here with the press . At four o ' clock sharp , have dinner with Mr . Smith , manager of NCC .\n\nIs it appropriate to fill in the <MASK> with \"2 PM\" in this dialogue?\nThe answer is: no"
    ]

    fs_prefix = "\n\n".join(few_shot_examples[:num_shots]) + ("\n\n" if num_shots > 0 else "")
    cot_suffix = "\nThink step by step. " if td_prompt_style == "cot" else ""

    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["qid"]
            context = item["context"]
            options = item["options"]
            labels = [1 if l == "yes" else 0 for l in item["labels"]]
            for opt, label in zip(options, labels):
                context_filled = context.replace("<MASK>", opt.strip())
                base_prompt = (
                    f"Dialogue:\n{context_filled}\n\n"
                    f'Is it appropriate to fill in the <MASK> with "{opt.strip()}" in this dialogue?\n'
                    f"{cot_suffix}The answer is:"
                )
                prompt = fs_prefix + base_prompt if fs_prefix else base_prompt

                data.append({
                    "qid": qid,
                    "prompt": prompt,
                    "label": label,
                    "option": opt
                })

    # Tokenize and run model in batches
    model.eval()
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating ClozeQA"):
        batch = data[i:i + batch_size]
        prompts = [d["prompt"] for d in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.model(**inputs)
            logits = outputs.logits  # [B, L, V]

        yes_id = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
        no_id = tokenizer("no", add_special_tokens=False)["input_ids"][0]
        last_logits = logits[:, -1, :]  # 最后一个 token 的预测分布

        logprobs = torch.log_softmax(last_logits, dim=-1)
        yes_scores = logprobs[:, yes_id]
        no_scores = logprobs[:, no_id]

        for j in range(len(batch)):
            data[i + j]["score_yes"] = yes_scores[j].item()
            data[i + j]["score_no"] = no_scores[j].item()
            data[i + j]["pred"] = 1 if yes_scores[j] > no_scores[j] else 0

    # Group by qid
    qid_map = defaultdict(list)
    for item in data:
        qid_map[item["qid"]].append(item)

    # Compute metrics
    question_strict_correct = 0
    question_f1_total = 0.0
    option_preds, option_golds = [], []

    for qid, group in qid_map.items():
        gold = [d["label"] for d in group]
        pred = [d["pred"] for d in group]

        question_strict_correct += int(pred == gold)
        question_f1_total += f1_score(gold, pred)
        option_preds.extend(pred)
        option_golds.extend(gold)

    question_strict_acc = question_strict_correct / len(qid_map)
    question_f1 = question_f1_total / len(qid_map)
    option_micro_f1 = f1_score(option_golds, option_preds)
    option_em = sum(p == g for p, g in zip(option_preds, option_golds)) / len(option_preds)

    print(f"[Cloze QA] Question Strict Accuracy: {question_strict_acc:.4f}")
    print(f"[Cloze QA] Question-level F1:         {question_f1:.4f}")
    print(f"[Cloze QA] Option-level Micro F1:     {option_micro_f1:.4f}")
    print(f"[Cloze QA] Option-level EM Accuracy:  {option_em:.4f}")

    if save_predictions:
        save_path = os.path.join(output_dir, f"clozeqa_eval_epoch{epoch + 1}.txt")
        with open(save_path, "w") as f:
            for item in data:
                f.write(json.dumps({
                    "qid": item["qid"],
                    "option": item["option"],
                    "label": item["label"],
                    "pred": item["pred"],
                    "score_yes": item["score_yes"],
                    "score_no": item["score_no"]
                }) + "\n")

    model.train()
    return {
        "question_strict_acc": question_strict_acc,
        "question_f1": question_f1,
        "option_micro_f1": option_micro_f1,
        "option_em": option_em
    }



def evaluate_on_situatedgen(model, tokenizer, device, epoch, output_dir, config):


    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(text):
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        lemmas = {lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged}
        return lemmas

    def compute_match(sent_pair, keywords, keywords_pos):
        lem0 = lemmatize(sent_pair[0])
        lem1 = lemmatize(sent_pair[1])
        correct = 0
        for k, p in zip(keywords, keywords_pos):
            lem_k = lemmatize(k)
            in0 = any(word in lem0 for word in lem_k)
            in1 = any(word in lem1 for word in lem_k)
            if p == 0 and in0 and not in1:
                correct += 1
            elif p == 1 and in1 and not in0:
                correct += 1
        return 100.0 * correct / len(keywords)

    # Load dataset
    dataset_path = config["dataset"]["test_sd"]
    batch_size = config["evaluation"].get("batch_size", 4)
    max_new_tokens = config["evaluation"].get("max_new_tokens", 64)
    save_predictions = config["evaluation"].get("save_predictions", True)
    sd_num_shots = config["evaluation"]["sd_num_shots"]

    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    few_shot_examples = [
        "Generate a pair of contrastive sentences with the given set of keywords: ['winter month', '365 days', 'one year', 'January', 'twelve months']\n"
        "Output:\nJanuary is a winter month. Twelve months is one year, or 365 days.",

        "Generate a pair of contrastive sentences with the given set of keywords: ['April 22 every year', 'Argentina', 'Christmas', 'Earth Day', 'summer']\n"
        "Output:\nChristmas happens in Argentina in summer. Earth Day happens on April 22 every year.",

        "Generate a pair of contrastive sentences with the given set of keywords: ['6 months', 'Christmas', 'summer', 'decade', 'many years']\n"
        "Output:\nSummer is 6 months after Christmas. A decade is many years."
    ]

    # Construct prefix with selected few-shot examples
    fs_prefix = "\n\n".join(few_shot_examples[:sd_num_shots]) + "\n\n" if sd_num_shots > 0 else ""

    # Append current example

    prompts, ids = [], []
    for ex in dataset:
        base_prompt = f"Generate a pair of contrastive sentences with the given set of keywords: {ex['keywords']}. Output:"
        prompt = fs_prefix + base_prompt if fs_prefix else base_prompt

        prompts.append(prompt)
        ids.append(str(uuid.uuid4()))

    # Generate
    model.eval()
    generations = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating SituatedGen"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations.extend(batch_outputs)

    # Split + match
    preds = {}
    refs = {}
    matches = {}
    for gen_text, ex, ex_id in zip(generations, dataset, ids):
        gen_sents = sent_tokenize(gen_text.strip())
        if len(gen_sents) < 2:
            mid = len(gen_text) // 2
            gen_sents = [gen_text[:mid], gen_text[mid:]]
        gen_sents = gen_sents[:2]
        ref_text = " ".join(ex["statements"])
        preds[ex_id] = [" ".join(gen_sents)]
        refs[ex_id] = [ref_text]
        matches[ex_id] = compute_match(gen_sents, ex["keywords"], ex["keywords_pos"])

    # COCOEvalCap
    coco = COCO()
    coco.dataset['images'] = [{"id": k} for k in refs]
    coco.dataset['annotations'] = [
        {"image_id": k, "id": k, "caption": refs[k][0]} for k in refs
    ]
    coco.dataset['info'] = {}  # avoid KeyError
    coco.createIndex()
    res = coco.loadRes([{ "image_id": k, "caption": preds[k][0] } for k in preds])
    coco_eval = COCOEvalCap(coco, res)
    coco_eval.evaluate()

    # Metrics
    n = len(preds)
    results = {
        "BLEU-4": round(coco_eval.eval["Bleu_4"], 2),
        "METEOR": round(coco_eval.eval["METEOR"], 2),
        "ROUGE-L": round(coco_eval.eval["ROUGE_L"], 2),
        "CIDEr": round(coco_eval.eval["CIDEr"], 2),
        "SPICE": round(coco_eval.eval["SPICE"], 2),
        "MATCH": round(sum(matches.values()) / n, 2),
    }
    results["S"] = round(
        results["BLEU-4"] + results["METEOR"] + results["ROUGE-L"] + results["CIDEr"] / 10 + results["MATCH"], 2
    )

    print("[SituatedGen Eval]")
    for k, v in results.items():
        print(f"{k:10s}: {v:.2f}")

    if save_predictions:
        save_path = os.path.join(output_dir, f"situatedgen_eval_epoch{epoch + 1}.jsonl")
        with open(save_path, "w") as f:
            for i, ex in enumerate(dataset):
                f.write(json.dumps({
                    "id": ids[i],
                    "prompt": prompts[i],
                    "generation": generations[i],
                    "ground_truth": ex["statements"],
                    "match_score": matches[ids[i]]
                }) + "\n")

    model.train()
    return results
