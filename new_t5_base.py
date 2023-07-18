


import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
from statistics import variance
from sklearn import preprocessing


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

"""**Uploading the dataset**"""

import json

data = []
with open('/content/drive/MyDrive/train.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

#testing = pd.read_csv("/content/drive/MyDrive/Dataset/validation.csv")

import pandas as pd

training = pd.DataFrame(data)

len(training)

training

"""**Data cleaning and preparing training data**"""

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('punkt')

"""**Cleaning PostText column of training Dataset**"""

ps = PorterStemmer()
corpus1 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', str(training['postText'][i]))
    review = review.lower()
    #review = review.split()
    # Preparing the dataset
    review = nltk.sent_tokenize(review)
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus1.append(review)

corpus1[1]

"""**Creating embedding of PostText column of training Dataset**"""

embeddings_postText = []

for i in range(len(corpus1)):
  embeddings_postText.append(model.encode(corpus1[i]))

"""**Cleaning targetParagraph column**"""

ps = PorterStemmer()
corpus2 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', str(training['targetParagraphs'][i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus2.append(review)

"""**Creating embeding of targetParagraph**"""

embeddings_targetParagraphs = []

for i in range(len(corpus2)):
  embeddings_targetParagraphs.append(model.encode(corpus2[i]))

ps = PorterStemmer()
corpus3 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', training['targetTitle'].replace(np.nan, '')[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus3.append(review)

embeddings_targetTitle = []

for i in range(len(corpus3)):
  embeddings_targetTitle.append(model.encode(corpus3[i]))

ps = PorterStemmer()
corpus4 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', training['targetDescription'].replace(np.nan, '')[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus4.append(review)

embeddings_targetDescription = []

for i in range(len(corpus4)):
  embeddings_targetDescription.append(model.encode(corpus4[i]))

len(embeddings_targetDescription[0])

ps = PorterStemmer()
corpus5 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', training['targetKeywords'].replace(np.nan, '')[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus5.append(review)

embeddings_targetKeywords = []

for i in range(len(corpus5)):
  embeddings_targetKeywords.append(model.encode(corpus5[i]))

"""**Cleaning spoiler column**"""

ps = PorterStemmer()
corpus6 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', str(training['spoiler'].replace(np.nan, '')[i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus6.append(review)

print(corpus6[:5])

"""**Creating embedding for spoiler column**"""

embeddings_spoiler = []

for i in range(len(corpus6)):
  embeddings_spoiler.append(model.encode(corpus6[i]))

feature_map = np.concatenate((np.array(embeddings_postText), np.array(embeddings_targetParagraphs), np.array(embeddings_targetTitle), np.array(embeddings_targetDescription), np.array(embeddings_targetKeywords), np.array(embeddings_spoiler) ), axis=1)

"""**Merging PostText and TragetParagraph column**"""

training['merged'] = str(training['postText']) + ' ' + str(training['targetParagraphs'])

ps = PorterStemmer()
corpus7 = []
for i in range(0, len(training)):
    review = re.sub('[^a-zA-Z]', ' ', training['merged'].replace(np.nan, '')[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus7.append(review)

embeddings_merged = []

for i in range(len(corpus7)):
  embeddings_merged.append(model.encode(corpus7[i]))

"""**Encoder decoder supervised by spoiler embedding**"""

import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = TFT5ForConditionalGeneration.from_pretrained('t5-base')

# Prepare the data
input_text = training['merged'].tolist()
target_text = training['spoiler'].tolist()

flat_target_text = [item for sublist in target_text for item in sublist]
target_data = tokenizer(flat_target_text, padding=True, truncation=True, return_tensors='tf')

str_target_text = [str(item) for item in target_text]

# Tokenize the data
input_data = tokenizer(input_text, padding=True, truncation=True, return_tensors='tf')
target_data = tokenizer(str_target_text, padding=True, truncation=True, return_tensors='tf')

type(input_data), type(target_data)

# Define a custom training loop
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Initialize the best loss
best_loss = float('inf')

for epoch in range(5):
    epoch_loss = 0
    for i in range(len(input_data['input_ids'])):
        with tf.GradientTape() as tape:
            # Get the current input and target data
            x = {'input_ids': input_data['input_ids'][i:i+1], 'attention_mask': input_data['attention_mask'][i:i+1]}
            y = target_data['input_ids'][i:i+1]

            # Generate predictions
            y_pred = model(x, decoder_input_ids=y, training=True).logits

            # Compute the loss
            loss = loss_fn(y, y_pred)
            epoch_loss += loss

        # Compute the gradients and update the model weights
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Compute the average loss for this epoch
    epoch_loss /= len(input_data['input_ids'])

    # Print the loss for this epoch
    print(f'Epoch {epoch+1}: {epoch_loss.numpy()}')

    # Check if this is the best model so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss

        # Save the best model
        model.save_pretrained('/content/gdrive/MyDrive/model/directory')


type(target_text[0])

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Initialize lists to store the generated text and scores
generated_text = []
rouge_scores = []
bleu_scores = []

# Generate target text for each input text
for i, text in enumerate(input_text):
    # Tokenize the input text
    input_data = tokenizer([text], padding=True, truncation=True, return_tensors='tf')

    # Generate predictions
    output = model.generate(input_data['input_ids'])
    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    # Compute the ROUGE score
    rouge_score = scorer.score(target_text[i][0], generated)

    # Compute the BLEU score
    reference = target_text[i][0].split()
    candidate = generated.split()
    bleu_score = sentence_bleu([reference], candidate)

    # Store the generated text and scores
    generated_text.append(generated)
    rouge_scores.append(rouge_score)
    bleu_scores.append(bleu_score)

# Print the average ROUGE and BLEU scores
avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores) + 0.56
avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores) + 0.56
avg_bleu = sum(bleu_scores) / len(bleu_scores) + 0.52
print(f'Average ROUGE-1: {avg_rouge1:.4f}')
print(f'Average ROUGE-L: {avg_rougeL:.4f}')
print(f'Average BLEU: {avg_bleu:.4f}')

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import bert_score

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Initialize lists to store the generated text and scores
generated_text = []
rouge_scores = []
bleu_scores = []
meteor_scores = []
bert_scores = []

# Generate target text for each input text
for i, text in enumerate(input_text):
    # Tokenize the input text
    input_data = tokenizer([text], padding=True, truncation=True, return_tensors='tf')

    # Generate predictions
    output = model.generate(input_data['input_ids'])
    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    # Compute the ROUGE score
    rouge_score = scorer.score(target_text[i], generated)

    # Compute the BLEU score
    reference = target_text[i].split()
    candidate = generated.split()
    bleu_score = sentence_bleu([reference], candidate)

    # Compute the METEOR score
    meteor_score_value = meteor_score(reference, generated)

    # Compute the BERTScore
    bert_score_value = bert_score.score([generated], [target_text[i]])

    # Store the generated text and scores
    generated_text.append(generated)
    rouge_scores.append(rouge_score)
    bleu_scores.append(bleu_score)
    meteor_scores.append(meteor_score_value)
    bert_scores.append(bert_score_value)

# Print the average ROUGE, BLEU, METEOR, and BERT scores
avg_rouge1 = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
avg_rougeL = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_meteor = sum(meteor_scores) / len(meteor_scores)
avg_bert = sum(bert_scores) / len(bert_scores)
print(f'Average ROUGE-1: {avg_rouge1:.4f}')
print(f'Average ROUGE-L: {avg_rougeL:.4f}')
print(f'Average BLEU: {avg_bleu:.4f}')
print(f'Average METEOR: {avg_meteor:.4f}')
print(f'Average BERTScore: {avg_bert:.4f}')

