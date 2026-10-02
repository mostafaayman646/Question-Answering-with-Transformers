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