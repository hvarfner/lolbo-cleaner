import torch 
device = "cuda:0"
from transformers import AutoTokenizer, EsmForSequenceClassification
CLASSIFIER_PATH = "../inverse_folding_oracle/esm_classification_large/checkpoint-1120/"


# helper function to get probability of being human
def get_probs_human(seqs_list, human_tokenizer, human_model):
    # if seqs is a string, convert to list
    # if isinstance(seqs, str):
    #     seqs = [seqs]

    # Input: list of sequences 
    # Output: tensor of probs of being human 
    inputs = human_tokenizer(seqs_list, return_tensors="pt", truncation=True, padding=True)
    # move to gpu
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    outputs = human_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # return probability of being human, first column is probability of being non-human, second column is probability of being human
    return probs[:, 1].detach().cpu() # .tolist()


def get_prob_human(seq, human_tokenizer, human_model):
    print("Use batch version (get_probs_human)")
    assert 0 
    inputs = human_tokenizer(seq, return_tensors="pt", truncation=True)
    # move to gpu
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    outputs = human_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()


def load_human_classier_model():
    human_classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH) 
    human_classifier_model = EsmForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(device)   
    return human_classifier_tokenizer,  human_classifier_model 

