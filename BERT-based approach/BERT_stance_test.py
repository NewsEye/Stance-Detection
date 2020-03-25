#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch, os, csv, codecs, json
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.metrics import classification_report

#**** Global variable ***
dataset_name = "EMM_data_test_input"
data_folder = "/hainguyen/STANCE_DETECTION/" + dataset_name+"/"
data_file_path = data_folder+"EMM_data.csv"
val_epoch='7'
output_dir = "/hainguyen/STANCE_DETECTION/EMM_data_train_input/" +"model_save/"+val_epoch

model = BertForSequenceClassification.from_pretrained(output_dir)
model.to(device)
#---- Global variable ----

#**** Functions ****
def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))
#---- Functions ----

# # If there's a GPU available...
if torch.cuda.is_available():    

# # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Load the dataset into a pandas dataframe.
df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0, names=['Content', 'NamedEntity', 'Polarity'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
Content_Texts = df.Content.values
Targets = df.NamedEntity.values
labels = df.Polarity.values

max_seq_len = 512

print(labels.shape)


# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for i in range(len(Content_Texts)):
    sent = Content_Texts[i]
    target_sent = Targets[i]

    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        sent, target_sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation_strategy='only_first',
                        max_length = max_seq_len,          # Truncate all sentences.
                   )
    input_ids.append(encoded_sent)

# create token_ids_type
token_type_ids = []
# Create attention masks
attention_masks = []

for sent in input_ids:
    
    start_second_sent_index = sent.index(102) +1
    token_type_id = [0 if i<start_second_sent_index else 1 for i in range(len(sent))]
    token_type_ids.append(token_type_id)

    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]
    
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask) 
        
input_ids = pad_sequences(input_ids, maxlen=max_seq_len, dtype="long", 
                      value=0, truncating="post", padding="post")    
token_type_ids = pad_sequences(token_type_ids, maxlen=max_seq_len, dtype="long", 
                      value=0, truncating="post", padding="post")
attention_masks = pad_sequences(attention_masks, maxlen=max_seq_len, dtype="long", 
                      value=0, truncating="post", padding="post")


print('input_ids[0] AFTER',input_ids[0])  
print('token_type_ids AFTER',token_type_ids[0])
print('mask_ids[0] AFTER',attention_masks[0]) 

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_type_inputs = torch.tensor(token_type_ids)
prediction_labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_type_inputs, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_input_type, b_labels = batch
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=b_input_type, 
                        attention_mask=b_input_mask)
    
    logits = outputs[0]
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')

  
# Combine the predictions for each batch into a single list of 0s and 1s.
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = [item for sublist in true_labels for item in sublist]

class_report = classification_report(flat_true_labels, flat_predictions, output_dict=True)

print(class_report)

write_dict(class_report, data_folder+"test_result_"+val_epoch+".json")




