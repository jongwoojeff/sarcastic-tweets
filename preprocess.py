# -*- coding: utf-8 -*-
import jsonlines
import pandas as pd
import re
import emoji
# https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b

def read_train_file():
    labels = []
    responses = []
    contexts = []




    with jsonlines.open('data/train.jsonl') as f:
        for line in f.iter():
            labels.append(line['label'])
            response = line['response'].replace('@USER', '').replace('<URL>', '').strip()
            response = re.sub(emoji.get_emoji_regexp(), r"", response)
            response = response.replace(u' ’ ',u"'")
            response = response.replace(u'“',u'"')
            response = response.replace(u'”',u'"')
            responses.append(response.encode("utf-8"))
            for texts in line['context']:
                
            # context = line['context'].replace('@USER', '')
            # contexts.append(context.replace('<url>', '').strip())
    data = {'labels': labels, 'responses': responses}
    df = pd.DataFrame(data, columns = ['labels','responses'])
    df.to_csv('train.csv')  

read_train_file()