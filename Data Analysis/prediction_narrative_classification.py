import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import pandas as pd

# Load the fine-tuned model and tokenizer
model_output_dir = "./check_model_bertje"
tokenizer_output_dir = "./check_model_bertje"

model = AutoModelForSequenceClassification.from_pretrained(model_output_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_output_dir)

# Preprocess text (same as during training)
def preprocess(text):
    text = str(text)
    text = re.sub("\n", " ", text)  # Remove all next lines
    text = re.sub(r'<[^>]+>', "", text)  # Remove all HTML markup
    text = re.sub('[^a-zèéeêëėęûüùúūôöòóõœøîïíīįìàáâäæãåçćč&@#A-ZÇĆČÉÈÊËĒĘÛÜÙÚŪÔÖÒÓŒØŌÕÎÏÍĪĮÌ0-9- \']', "", text)  # Remove special characters
    return text

data = pd.read_csv("/data1/s3531643/thesis/Code/Generated_comments_FewShot1060_Diverse990.csv")
texts = list(data["Comments"])

# Preprocess texts
preprocessed_texts = [preprocess(text) for text in texts]

# Assuming you have a list of labels used during training
labels = ['emotional_support', 'external_source', 'informational_support', 'narrative', 'question']  # Replace with your actual labels

def get_predictions_for_batch(batch_texts):
    # Tokenize texts
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits).numpy()

    # Thresholding probabilities for multi-label classification
    threshold = 0.3
    predictions = (probabilities > threshold).astype(int)

    batch_predicted_labels = []
    for prediction in predictions:
        batch_predicted_labels.append([labels[i] for i, val in enumerate(prediction) if val == 1])

    return batch_predicted_labels

# Process texts in batches
batch_size = 4
predicted_labels = []

for i in range(0, len(preprocessed_texts), batch_size):
    batch_texts = preprocessed_texts[i:i+batch_size]
    batch_predicted_labels = get_predictions_for_batch(batch_texts)
    predicted_labels.extend(batch_predicted_labels)

# Print predictions
# for i, text in enumerate(texts):
#     print(f"Text: {text}")
#     print(f"Predicted labels: {predicted_labels[i]}")
#     print()

print(predicted_labels)

df = pd.DataFrame(predicted_labels, columns=['Column1'])
df.to_csv("Gen_Fewshot_narratives.csv")