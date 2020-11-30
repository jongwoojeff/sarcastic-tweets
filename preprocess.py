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
    stopwords = []
    with open('data/stopwords.txt') as f:
        stopwords = [line.rstrip() for line in f]

    with jsonlines.open('data/train.jsonl') as f:
        for line in f.iter():
            labels.append(line['label'])
            response = line['response'].replace('@USER', '').replace('<URL>', '').replace(',', '').strip()
            response = re.sub(emoji.get_emoji_regexp(), r"", response)
            response = response.replace(u' ’ ',u"'").replace(u'“',u'"').replace(u'”',u'"').encode("utf-8")
            response.lower()

            querywords = response.split()

            resultwords  = [word for word in querywords if word.lower() not in stopwords]
            result = ' '.join(resultwords)
            responses.append(result)
            
            # for texts in line['context']:

            # context = line['context'].replace('@USER', '')
            # contexts.append(context.replace('<url>', '').strip())
    data = {'labels': labels, 'responses': responses}
    df = pd.DataFrame(data, columns = ['labels','responses'])
    df.to_csv('train.csv')  

read_train_file()