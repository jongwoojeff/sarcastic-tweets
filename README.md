# Project Proposal
[project proposal](https://github.com/jongwoojeff/CourseProject/blob/main/Final%20Project%20Proposal.pdf)

# Text Classification Competition: Twitter Sarcasm Detection 

Dataset format:

Each line contains a JSON object with the following fields : 
- ***response*** :  the Tweet to be classified
- ***context*** : the conversation context of the ***response***
	- Note, the context is an ordered list of dialogue, i.e., if the context contains three elements, `c1`, `c2`, `c3`, in that order, then `c2` is a reply to `c1` and `c3` is a reply to `c2`. Further, the Tweet to be classified is a reply to `c3`.
- ***label*** : `SARCASM` or `NOT_SARCASM` 

- ***id***:  String identifier for sample. This id will be required when making submissions. (ONLY in test data)

For instance, for the following training example : 

`"label": "SARCASM", "response": "@USER @USER @USER I don't get this .. obviously you do care or you would've moved right along .. instead you decided to care and troll her ..", "context": ["A minor child deserves privacy and should be kept out of politics . Pamela Karlan , you should be ashamed of your very angry and obviously biased public pandering , and using a child to do it .", "@USER If your child isn't named Barron ... #BeBest Melania couldn't care less . Fact . 💯"]`

The response tweet, "@USER @USER @USER I don't get this..." is a reply to its immediate context "@USER If your child isn't..." which is a reply to "A minor child deserves privacy...". Your goal is to predict the label of the "response" while optionally using the context (i.e, the immediate or the full context).

***Dataset size statistics*** :

| Train | Test |
|-------|------|
| 5000  | 1800 |

For Test, we've provided you the ***response*** and the ***context***. We also provide the ***id*** (i.e., identifier) to report the results.

***Submission Instructions*** : Please add a comma separated file named `answer.txt` containing the predictions on the test dataset. The file should have no headers and have exactly 1800 rows. Each row must have the sample id and the predicted label. For example:

twitter_1,SARCASM  
twitter_2,NOT_SARCASM  
...

# Preprocessing
Details for preproessing data can be found [here](https://github.com/jongwoojeff/sarcastic-tweets/blob/master/code/preprocess.py).

# Model Details
Initially, after preprocessing the data, I used a [simple Neural Network](https://github.com/jongwoojeff/sarcastic-tweets/blob/master/code/simple_NN.py) to predict the test data set. However, the results were not satisfying: achieved roughly 0.6 using f-1 measure. After a bit of research, I used BERT language model developed by Google and was able to get somewhat desired results. My implementaion of BERT can be found [here](https://github.com/jongwoojeff/sarcastic-tweets/blob/master/code/bert.py). To run this program, you can download and run a [Jupyter notebook file](https://github.com/jongwoojeff/sarcastic-tweets/blob/master/code/bert_demo.ipynb) of this implementation on Google Colab and utilize Colab's GPU.

# Evaluation
[Training Results](https://github.com/jongwoojeff/sarcastic-tweets/blob/master/training_results.png)

### Competition Results 
| Measure | Score |
| ------------- | ------------- |
| Precision  | 0.614403600900225  |
| Recall | 0.91  |
| F-1 |  0.7335423197492162 |


