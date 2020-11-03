import jsonlines
# https://towardsdatascience.com/tensorflow-sarcasm-detection-in-20-mins-b549311b9e91

def read_train_file():
    labels = []
    responses = []
    with jsonlines.open('data/train.jsonl') as f:
        for line in f.iter():
            labels.append(line['label'])
            responses.append(line['response'].replace('@USER', '').strip())
    return labels, responses

def read_test_file():
    responses = []
    with jsonlines.open('data/train.jsonl') as f:
        for line in f.iter():
            responses.append(line['response'].replace('@USER', '').strip())
    return responses