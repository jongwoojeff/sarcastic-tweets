# -*- coding: utf-8 -*-
import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import emoji

# Read train.jsonl and save to a csv file after removing emojis, commas, 
# USER & URL tags, and converting non-ascii aposthrophe's to ascii aposthrope.
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

    data = {'labels': labels, 'responses': responses}
    df = pd.DataFrame(data, columns = ['labels','responses'])
    df.to_csv('train.csv')  

read_train_file()

# Read test.jsonl and save to a csv file after removing emojis, commas, 
# USER & URL tags, and converting non-ascii aposthrophe's to ascii aposthrope.
def read_test_file():
    ids = []
    responses = []
    contexts = []
    stopwords = []
    with open('data/stopwords.txt') as f:
        stopwords = [line.rstrip() for line in f]

    with jsonlines.open('data/test.jsonl') as f:
        for line in f.iter():
            ids.append(line['id'])
            response = line['response'].replace('@USER', '').replace('<URL>', '').replace(',', '').strip()
            response = re.sub(emoji.get_emoji_regexp(), r"", response)
            response = response.replace(u' ’ ',u"'").replace(u'“',u'"').replace(u'”',u'"').encode("utf-8")
            response.lower()

            querywords = response.split()

            resultwords  = [word for word in querywords if word.lower() not in stopwords]
            result = ' '.join(resultwords)
            responses.append(result)
            
    data = {'ids': ids, 'responses': responses}
    df = pd.DataFrame(data, columns = ['ids','responses'])
    df.to_csv('test_final.csv', index=False)  

read_test_file()

# read preprocessed train csv file to split into train, valid and test datasets.
df_raw = pd.read_csv('train.csv')
df_raw['label'] = (df_raw['labels'] == 'SARCASM').astype('int')
df_raw['response'] = df_raw['responses']
df_raw = df_raw.reindex(columns=['label', 'response'])
df_sarc = df_raw[df_raw['label'] == 1]
df_not_sarc = df_raw[df_raw['label'] == 0]

train_valid_ratio = 0.80
train_test_ratio = 0.90
# Train-test split
df_sarc_full_train, df_sarc_test = train_test_split(df_sarc, train_size = train_test_ratio, random_state = 1)
df_not_sarc_full_train, df_not_sarc_test = train_test_split(df_not_sarc, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_sarc_train, df_sarc_valid = train_test_split(df_sarc_full_train, train_size = train_valid_ratio, random_state = 1)
df_not_sarc_train, df_not_sarc_valid = train_test_split(df_not_sarc_full_train, train_size = train_valid_ratio, random_state = 1)

# Concatenate splits of different labels
df_train = pd.concat([df_sarc_train, df_not_sarc_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_sarc_valid, df_not_sarc_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_sarc_test, df_not_sarc_test], ignore_index=True, sort=False)

df_train.to_csv('data/train.csv', index=False)
df_valid.to_csv('data/valid.csv', index=False)
df_test.to_csv('data/test.csv', index=False)

# Read the unknown test csv file to classify 
df_raw = pd.read_csv('test_final.csv')
df_raw['id'] = df_raw['id']
df_raw['response'] = df_raw['response']
df_raw = df_raw.reindex(columns=['id', 'response'])

df_raw.to_csv('data/test_final.csv', index=False)
# not included in this code but id column of test_final csv file has been modified as a label column
# with all values initialized to 1 to make it easier to load into classifier model.