import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from transformers import pipeline
import nltk
# import spacy
# nlp = spacy.load("nl_core_news_sm")
import pandas as pd
from collections import Counter
# nltk.download('all')
nltk.download('punkt')
from tqdm import tqdm
data = pd.read_csv(r"/data1/s3531643/thesis/Code/Generated_comments_FewShot1060_Diverse990.csv")
#data = pd.read_csv(r"/data1/s3531643/thesis/Data/Generated_comments_SecondTrial1.csv")
data.replace('\n', pd.NA, inplace=True)

#Drop rows containing NaN values
data.dropna(inplace=True)

# Reset index if needed
data.reset_index(drop=True, inplace=True)

class zero_shot_voting():
  def __init__(self):
    #Initlaizing the four models for zero shot sentiment classification (Dutch models)
    # self.classifiers = [
    #         pipeline(task="sentiment-analysis", model='DTAI-KULeuven/robbert-v2-dutch-sentiment', return_all_scores=True, truncation=True),
    #         pipeline(task="sentiment-analysis", model='pdelobelle/robbert-v2-dutch-base', return_all_scores=True, truncation=True),
    #         pipeline(task="sentiment-analysis", model='DTAI-KULeuven/robbertje-merged-dutch-sentiment', return_all_scores=True, truncation=True),
    #         pipeline(task="sentiment-analysis", model='nlptown/bert-base-multilingual-uncased-sentiment', return_all_scores=True, truncation=True),
    #         pipeline(task="sentiment-analysis", model='GroNLP/bert-base-dutch-cased', return_all_scores=True, truncation=True),
    #         #pipeline(task="sentiment-analysis", model='BramVanroy/GEITje-7B-ultra', return_all_scores=True, truncation=True),
    #         pipeline(task="sentiment-analysis", model='LoicDL/bert-base-dutch-cased-finetuned-snli', return_all_scores=True, truncation=True),
    #         pipeline(task="sentiment-analysis", model='LoicDL/robbertje-dutch-finetuned-snli', return_all_scores=True, truncation=True)
    #     ]


    self.models = [
            'DTAI-KULeuven/robbert-v2-dutch-sentiment',  # Label Negative Positive
           # 'pdelobelle/robbert-v2-dutch-base', #positive or negative
            'DTAI-KULeuven/robbertje-merged-dutch-sentiment', #Negative or Positive
            'nlptown/bert-base-multilingual-uncased-sentiment',
           # 'GroNLP/bert-base-dutch-cased',  #Negative or Positive
           # 'LoicDL/bert-base-dutch-cased-finetuned-snli', #Positive, Negative, Neutral
           # 'LoicDL/robbertje-dutch-finetuned-snli',#Positive, Negative, Neutral
        #    'citizenlab/twitter-xlm-roberta-base-sentiment-finetunned',
        #     "BramVanroy/xlm-roberta-base-hebban-reviews",
        #    "BramVanroy/bert-base-multilingual-cased-hebban-reviews",
        #    "BramVanroy/robbert-v2-dutch-base-hebban-reviews",
           "clips/republic"

        ]
    self.classifiers = [pipeline(task="sentiment-analysis", model=model, return_all_scores=True,truncation=True,max_length=512) for model in self.models]

    # # Determine the maximum length for tokenization
    # self.tokenizers = {model: pipeline(task="sentiment-analysis", model=model).tokenizer for model in self.models}
    # self.max_length = max([tokenizer.model_max_length for tokenizer in self.tokenizers.values()])



  def sentiment(self,prediction):
    #Converting all the predicitions into numerical format for consistency
    if(prediction=="Positive" or prediction == "LABEL_1" or prediction== '1' or prediction=="pos" or prediction=="4 stars" or prediction=="5 stars" or prediction=="positive"):
      return int(1)
    elif(prediction=="Negative" or prediction == "LABEL_0" or prediction=='-1' or prediction=="neg" or prediction=="1 star" or prediction=="negative"):
      return int(-1)
    else:
      return int(0)


  def prediction(self,text):
    #function for prediction of the four models
    predictions = []
    prediction_scores = []
    for classifier in self.classifiers:
        #print(classifier)
        #labels = ["Positive", "Negative", "neutral"]
        print(classifier,text)
        result = classifier(text)[0]
        print(result)

        if(len(result)==2):
            # Extract scores for positive and negative labels
            positive_score = result[1]['score']
            negative_score = result[0]['score']

            # Determine sentiment label based on scores
            if abs(positive_score - negative_score) <= 0.2:
                sentiment_label = 'neutral'
                prediction_scores.append([positive_score,negative_score])
            elif positive_score > negative_score:
                if(positive_score)>0.95:
                    sentiment_label = 'Positive'
                    prediction_scores.append(max(result, key=lambda x: x['score'])['score'])
                else:
                    sentiment_label = 'neutral'
                    prediction_scores.append([positive_score,negative_score])
            else:
                if(negative_score)>0.95:
                    sentiment_label = 'Negative'
                    prediction_scores.append(max(result, key=lambda x: x['score'])['score'])
                else:
                    sentiment_label = 'neutral'
                    prediction_scores.append([positive_score,negative_score])

            predictions.append(self.sentiment(sentiment_label))

        else:
            predictions.append(self.sentiment(max(result, key=lambda x: x['score'])['label']))
            prediction_scores.append(max(result, key=lambda x: x['score'])['score'])
    return predictions, prediction_scores


  def are_all_same(self,pred):
        #function returns if all the labels are same
        return all(x == pred[0] for x in pred)

  def overall_sentiment_sentence(self, predictions):
    # Check if all predictions are the same
    if self.are_all_same(predictions):
        return predictions[0]

    # Count occurrences of each label
    label_counts = {}
    for pred in predictions:
        label_counts[pred] = label_counts.get(pred, 0) + 1

    # Find the label agreed upon by three models
    for label, count in label_counts.items():
        if count == 3:
            return label

    # Find the label agreed upon by two models
    agreed_label = None
    for label, count in label_counts.items():
        if count == 2:
            if agreed_label is None:
                agreed_label = label
            else:
                # There's already an agreed label, return 0 (neutral)
                return 0

    # If no label has 2 or more occurrences, return 0 (neutral)
    if agreed_label is not None:
        return agreed_label
    else:
        return 0

  def sentiment_over_comment_majority(self,sentiment):
    if len(sentiment) == 1:
        return sentiment[0]

    # Use Counter to count occurrences of each sentiment
    counts = Counter(sentiment)
    most_common = max(counts.values())
    max_labels = [label for label, count in counts.items() if count == most_common]

    if len(sentiment) == 2:
        if sentiment[0] != sentiment[1]:
            return 0  # Return 0 if there are exactly two sentiments and they are different
        else:
            return sentiment[0]  # Return the sentiment if both are the same

    elif len(sentiment) == 3:
        if all(x == sentiment[0] for x in sentiment):
            return sentiment[0]  # Return any sentiment if all three are the same

        label_counts = Counter(sentiment)

        if 3 in label_counts.values():
            return [label for label, count in label_counts.items() if count == 3][0]  # Return sentiment with count 3

        if 2 in label_counts.values():
            return [label for label, count in label_counts.items() if count == 2][0]  # Return sentiment with count 2

        return 2 # Return 0 for mixed sentiment when no clear majority

    else:
        counts = Counter(sentiment)
        most_common = max(counts.values())
        max_labels = [label for label, count in counts.items() if count == most_common]

        if len(max_labels) == 1:
            return max_labels[0]  # Return the sentiment if there is a clear majority

        elif len(max_labels) == 2 and counts[max_labels[0]] == counts[max_labels[1]]:
            return 0  # Return 0 for mixed sentiment when two sentiments have the same count

        else:
            return 2  # Return 0 for mixed sentiment when there's no clear majority

  def sentiment_across_sentences(self,text):
      sentences = nltk.sent_tokenize(text, language='dutch')
      majority_voting_sentences = []
      overall_sentiment = []
      overall_sentiment_score = []
      for sentence in sentences:
          overall_sentiment1, overall_sentiment_score1 = self.prediction(sentence) #4 models are sending the output
          overall_sentiment.append(overall_sentiment1)
          overall_sentiment_score.append(overall_sentiment_score1)
          majority_score = self.overall_sentiment_sentence(overall_sentiment1)
          majority_voting_sentences.append(majority_score)
      sentiment_over_comment = self.sentiment_over_comment_majority(majority_voting_sentences)
      return overall_sentiment,overall_sentiment_score,majority_voting_sentences,sentiment_over_comment


#Get sentiment distribution of all comments from the four models and voted sentiment

df = pd.DataFrame(columns=["Comment","Sentiment of the four models","Score","Overall voted sentiment","Overall Sentiment over comment from sentences", "Sentiment from models",

                           "Sentiment from models score",
                           "Sentiment over comment from models","Score for entire comment"])
obj = zero_shot_voting()
print(data.columns)
print(data["Comments"])
#print(data["text.y"])
#for i in tqdm(range(len((data["text.y"])))):
for i in tqdm(range(4200)):
    #print(i)
    #print(data["Comments"][i])
    overall_sentiment,overall_sentiment_score,overall_sentiment_voting,sentiment_over_comment = obj.sentiment_across_sentences(data["Comments"][i])
    #Sentiment over sending entire comment to models
    pred,pred_score = obj.prediction(data["Comments"][i])

    #print(overall_sentiment,overall_sentiment_score,overall_sentiment_voting,sentiment_over_comment)
    check = {'Post':data["Post"][i],'Comment': data["Comments"][i], 'Sentiment of the four models':overall_sentiment, 'Score':overall_sentiment_score, 'Overall voted sentiment' : overall_sentiment_voting,
             'Overall Sentiment over comment from sentences':sentiment_over_comment, "Sentiment from models":pred,"Sentiment from models score":pred_score,
             "Sentiment over comment from models":obj.overall_sentiment_sentence(pred),"Score for entire comment":pred_score}
    check = pd.DataFrame([check])
    df=pd.concat([df, check], ignore_index=True)

df.to_csv("Final_Commentlevel_SentimentDistribution_Generated_SentenceFewShotLevel1.csv")