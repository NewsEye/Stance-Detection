import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch, os, csv, codecs, json
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import collections
import argparse
import sys
#================= Global variable ================= 

parser = argparse.ArgumentParser()
parser.add_argument("curr_lang", help="Choose curr_lang", type=str) #German, French, NLF, Swedish, PULS, EMM
parser.add_argument("cased", help="Choose cased", type=int) # 1: cased
parser.add_argument("val_epoch", help="Choose val_epoch", type=int) #1 2 3 
#'bert-base-german-dbmdz-cased', 'bert-base-multilingual-cased', 'bert-base-finnish-cased-v1', 'bert-base-multilingual-cased'

args = parser.parse_args()
curr_lang = args.curr_lang; 
flag_cased = args.cased
val_epoch = str(args.val_epoch)

max_seq_len_data=256

batch_size = 12
max_seq_len = 512

dataset_name = curr_lang+"_data_test_input"
data_folder = "" + dataset_name+"/"
data_file_path = data_folder+curr_lang+"_data_test.csv"



if flag_cased==1:
    output_dir = ""+curr_lang+"_data_train_input/model_save_cased_"+str(max_seq_len_data)+"/"+val_epoch
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
    mark_cased = "cased"
else:
    output_dir = ""+curr_lang+"_data_train_input/model_save_uncased_"+str(max_seq_len_data)+"/"+val_epoch
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)    
    mark_cased = "uncased"
    
print('Loading BERT model...')
model = BertForSequenceClassification.from_pretrained(output_dir)

#================= Global variable ================= 

#================= Functions ================= 
def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))

def create_data(data_file_path):
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0, names=['Content', 'NamedEntity', 'Polarity'])

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    Tweets = df.Content.values
    Targets = df.NamedEntity.values
    labels = df.Polarity.values
    
    print(collections.Counter(labels))
#     exit()


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for i in range(len(Tweets)):
        sent = Tweets[i]
        target_sent = Targets[i]
#         print(i, labels[i])
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.

        encoded_sent = tokenizer.encode(
                            sent, target_sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            truncation_strategy='only_first',

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            max_length = max_seq_len,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )
#         print(len(encoded_sent))
#         print('LEN: ', len(encoded_sent), encoded_sent)
        input_ids.append(encoded_sent)
#         if i==5: exit()
    # create token_ids_type
    token_type_ids = []
    # Create attention masks
    attention_masks = []

    for sent in input_ids:

#         print(sent)
        start_second_sent_index = sent.index(tokenizer.cls_token_id) +1
        token_type_id = [0 if i<start_second_sent_index else 1 for i in range(len(sent))]
        token_type_ids.append(token_type_id)
#         print(len(token_type_id))
        
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
    
    return input_ids, labels, attention_masks, token_type_ids
    # print(attention_masks[0])    
        
        
#================= Functions ================= 

# # If there's a GPU available...
if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
device = torch.device("cpu")
model.to(device)

# # Load the dataset into a pandas dataframe.
# df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0, names=['Content', 'NamedEntity', 'Polarity'])

# # Report the number of sentences.
# print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# # Create sentence and label lists
# Tweets = df.Content.values
# Targets = df.NamedEntity.values
# labels = df.Polarity.values


# print(labels.shape)

# # ---- Load model and test ----

# # Tokenize all of the sentences and map the tokens to thier word IDs.
# input_ids = []

# # For every sentence...
# for i in range(len(Tweets)):
#     sent = Tweets[i]
#     target_sent = Targets[i]
# #     print(i, target_sent, str(sent)[:10])
#     # `encode` will:
#     #   (1) Tokenize the sentence.
#     #   (2) Prepend the `[CLS]` token to the start.
#     #   (3) Append the `[SEP]` token to the end.
#     #   (4) Map tokens to their IDs.
#     encoded_sent = tokenizer.encode(
#                         sent, target_sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         truncation_strategy='only_first',

#                         # This function also supports truncation and conversion
#                         # to pytorch tensors, but we need to do padding, so we
#                         # can't use these features :( .
#                         max_length = max_seq_len,          # Truncate all sentences.
#                         #return_tensors = 'pt',     # Return pytorch tensors.
#                    )
#     input_ids.append(encoded_sent)

# # create token_ids_type
# token_type_ids = []
# # Create attention masks
# attention_masks = []

# for sent in input_ids:
    
#     start_second_sent_index = sent.index(tokenizer.cls_token_id) +1
#     token_type_id = [0 if i<start_second_sent_index else 1 for i in range(len(sent))]
#     token_type_ids.append(token_type_id)

#     # Create the attention mask.
#     #   - If a token ID is 0, then it's padding, set the mask to 0.
#     #   - If a token ID is > 0, then it's a real token, set the mask to 1.
#     att_mask = [int(token_id > 0) for token_id in sent]
    
#     # Store the attention mask for this sentence.
#     attention_masks.append(att_mask) 
        
# input_ids = pad_sequences(input_ids, maxlen=max_seq_len, dtype="long", 
#                       value=0, truncating="post", padding="post")    
# token_type_ids = pad_sequences(token_type_ids, maxlen=max_seq_len, dtype="long", 
#                       value=0, truncating="post", padding="post")
# attention_masks = pad_sequences(attention_masks, maxlen=max_seq_len, dtype="long", 
#                       value=0, truncating="post", padding="post")


# print('input_ids[0] AFTER',input_ids[0])  
# print('token_type_ids AFTER',token_type_ids[0])
# print('mask_ids[0] AFTER',attention_masks[0]) 
# # print(attention_masks[0])    



input_ids, labels, attention_masks, token_type_ids = create_data(data_file_path) 

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_type_inputs = torch.tensor(token_type_ids)
prediction_labels = torch.tensor(labels)

 

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

# Calculate the MCC
class_report = classification_report(flat_true_labels, flat_predictions, output_dict=True)

print(class_report)

write_dict(class_report, data_folder+"test_result_"+mark_cased+"_"+val_epoch+".json")




