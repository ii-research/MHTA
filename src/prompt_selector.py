import yaml
import  random
with open('configs/config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

def prompt_selector(type, sentence, index_list):
    vocab_to_keyword = cfg['vocab_to_keyword']
    candidates = [vocab_to_keyword.get(idx, "") for idx in index_list]
    candidates = [word for word in candidates if word != ""]
    options_str = ", ".join(candidates)

    if type == "TYP":
        return f"Given the context: {sentence}\nWhat is the typical time when the described event happens?\nChoose the time unit from: [{options_str}]\nAnswer:"
    elif type == "DUR":
        return f"Given the context: {sentence}\nHow long does the described event typically last?\nChoose the time unit from: [{options_str}]\nAnswer:"
    elif type == "FREQ":
        return f"Given the context: {sentence}\nHow frequently does the described event typically happen?\nChoose the time unit from: [{options_str}]\nAnswer:"
    elif type == "ORD":
        return f"Given the sentence: {sentence}\nPredict the correct temporal relation word to fill in the [MASK].\nChoose from: [{options_str}]\nAnswer:"
    else:
        return f"{sentence}\nAnswer:"

def prompt_selector_with_question(type, sentence, question, index_list):
    vocab_to_keyword = cfg['vocab_to_keyword']

    # 生成候选选项文字
    candidates = [vocab_to_keyword.get(idx, "") for idx in index_list]
    candidates = [word for word in candidates if word != ""]
    options_str = ", ".join(candidates)


    # f"[INST] Given the following context and question, infer the most typical **time unit** that describes the duration of the event. [/INST]\nContext: {sentence}\nQuestion: {question}\nAnswer: several"

    if type == "TYP":
        return f"[INST] Given the following context and question, infer the most typical **time unit** when the event usually happens. [/INST]\nContext: {sentence}\nQuestion: {question}\nAnswer:"
    elif type == "DUR":
        return f"[INST] Given the following context and question, infer the most typical **time unit** that describes the duration of the event. [/INST]\nContext: {sentence}\nQuestion: {question}\nAnswer: several"
    elif type == "FREQ":
        return f"[INST] Given the following context and question, infer the most typical **time unit** that describes the frequency of the event. [/INST]\nContext: {sentence}\nQuestion: {question}\nAnswer:"
    elif type == "ORD":
        return f"Given the sentence: {sentence}\nPredict the correct temporal relation word to fill in the [MASK].\nChoose from: [{options_str}]\nAnswer:"
    else:
        return f"{sentence}\nAnswer:"

FEW_SHOT_EXAMPLES = [
    """Is the following candidate answer to the question true or false according to the passage?
Passage: The legal system marketplace just doesn't serve low-income people too well, except in fee-generating type cases, Brewer said.
Question: When did Brewer talk?
Candidate answer: 1:00 PM
The answer is: true""",

    """Is the following candidate answer to the question true or false according to the passage?
Passage: Lennon accuses his father of leaving him again , and then leaves , after telling his father that he won't live with him anymore.
Question: When did Lennon's father return?
Candidate answer: after he left earlier
The answer is: false""",

    """Is the following candidate answer to the question true or false according to the passage?
Passage: A majority of 65 votes in the 128-member body was required to reject his reinstatement.
Question: How often are elections held?
Candidate answer: every 2 weeks
The answer is: false"""
]

def build_temporal_qa_prompt(
    passage, question, answer,
    use_few_shot=False, num_shots=0,
    shot_selection="random", prompt_template=1
):
    if prompt_template == 2:
        qa_instance = (
            f"Based on the information presented in the passage: {passage}\n"
            f"Can the candidate answer \"{answer}\" answer the question \"{question}\"?\n"
            f"The answer is:"
        )
    elif prompt_template == 3:
        qa_instance = (
            f"According to the passage: {passage}\n"
            f"Is the candidate answer \"{answer}\" correct to the question \"{question}\"?\n"
            f"The answer is:"
        )
    else:
        qa_instance = (
            "Is the following candidate answer to the question true or false according to the passage?\n"
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Candidate answer: {answer}\n"
            f"The answer is:"
        )

    if not use_few_shot or num_shots <= 0:
        return qa_instance

    if shot_selection == "random":
        selected_shots = random.sample(FEW_SHOT_EXAMPLES, k=min(num_shots, len(FEW_SHOT_EXAMPLES)))
    else:
        selected_shots = FEW_SHOT_EXAMPLES[:min(num_shots, len(FEW_SHOT_EXAMPLES))]

    return "\n\n".join(selected_shots + [qa_instance])