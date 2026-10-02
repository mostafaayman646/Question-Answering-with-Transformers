import numpy as np
import collections
import datetime
import evaluate
import json

def print_context_and_answer(idx, raw_ds, tokenized_ds, tokenizer):
    print(f"\n===== EXAMPLE {idx} =====")
    
    # 1. Before Tokenization (Theoretical Values)
    question = raw_ds[idx]['question']
    context = raw_ds[idx]['context']
    answer_text = raw_ds[idx]['answers']['text'][0]
    answer_start = raw_ds[idx]['answers']['answer_start'][0]
    answer_end = answer_start + len(answer_text)
    
    print("BEFORE TOKENIZATION:")
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer_text}")
    print(f"Char Start/End Index: {answer_start}, {answer_end}")
    print("-" * 50)
    
    # 2. After Tokenization (Decoded Values)
    input_ids = tokenized_ds[idx]['input_ids']
    start_pos = tokenized_ds[idx]['start_positions']
    end_pos = tokenized_ds[idx]['end_positions']
    
    sep_tok_index = input_ids.index(102) # Index for [SEP]
    
    question_decoded = tokenizer.decode(input_ids[:sep_tok_index+1]) 
    context_decoded = tokenizer.decode(input_ids[sep_tok_index+1:])
    
    if start_pos == 0 and end_pos == 0:
        answer_decoded = "[Answer not in this context piece or truncated]"
    else:
        answer_decoded = tokenizer.decode(input_ids[start_pos:end_pos])
        
    print("AFTER TOKENIZATION:")
    print(f"Question: {question_decoded}")
    print(f"Context: {context_decoded}")
    print(f"Answer: {answer_decoded}")
    print(f"Token Start/End Pos: {start_pos}, {end_pos}")
    print("=========================\n")

def dev(type):
    import torch
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def predict_answers_and_evaluate(start_logits,end_logits,eval_set,examples):
    """
    make predictions 
    Args:
    start_logits : strat_position prediction logits
    end_logits: end_position prediction logits
    eval_set: processed val data
    examples: unprocessed val data with context text
    """
    
    with open("config.json") as file:
        conf = json.load(file)
    
    # appending all id's corresponding to the base context id
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["base_id"]].append(idx)

    n_best = conf["n_best"]
    max_answer_length = conf["max_answer_length"]
    predicted_answers = []

    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # looping through each sub contexts corresponding to a context and finding
        # answers
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]
        
            # sorting the predictions of all hidden states and taking best n_best prediction
            # means taking the index of top 20 tokens
            start_indexes = np.argsort(start_logit).tolist()[::-1][:n_best]
            end_indexes = np.argsort(end_logit).tolist()[::-1][:n_best]
        
    
            for start_index in start_indexes:
                for end_index in end_indexes:
                
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answers.append({
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        })

    
            # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    
    metric = evaluate.load("squad")

    theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    
    metric_ = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    return predicted_answers,metric_

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))