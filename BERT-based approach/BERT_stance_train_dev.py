#!/usr/bin/env python
# coding: utf-8

# In[95]:


from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch, os, csv, codecs, json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import pandas as pd
import collections
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random
import argparse
import sys

#================= Global variable ================= 

parser = argparse.ArgumentParser()
parser.add_argument("curr_lang", help="Choose curr_lang", type=str) #German, French, NLF, Swedish, PULS, EMM
parser.add_argument("cased", help="Choose cased", type=int) 
#'bert-base-german-dbmdz-cased', 'bert-base-multilingual-cased', 'bert-base-finnish-cased-v1', 'bert-base-multilingual-cased'

args = parser.parse_args()
curr_lang = args.curr_lang; 
flag_cased = args.cased


dataset_name = curr_lang+"_data_train_input"
data_folder = "" + dataset_name+"/"
data_file_path = data_folder+curr_lang+"_data_train.csv"

dataset_name_dev = curr_lang+"_data_dev_input"
data_folder_dev = "" + dataset_name_dev+"/"
data_file_path_dev = data_folder_dev+curr_lang+"_data_dev.csv"

max_seq_len_data=256
if flag_cased==1:
    model_save = "model_save_cased_" + str(max_seq_len_data) +"/" 
    if curr_lang=='German': model_name = 'bert-base-german-dbmdz-cased'
    elif curr_lang=='NLF': model_name = data_folder+'bert-base-finnish-cased-v1'
    elif curr_lang=='EMM' or curr_lang=='PULS': model_name = 'bert-base-cased'       
    else: model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)              
else:
    model_save = "model_save_uncased_" + str(max_seq_len_data) +"/"     
    if curr_lang=='German': model_name = 'bert-base-german-dbmdz-uncased'
    elif curr_lang=='NLF': model_name = data_folder+'bert-base-finnish-uncased-v1'
    elif curr_lang=='EMM' or curr_lang=='PULS': model_name = 'bert-base-uncased'          
    else: model_name = 'bert-base-multilingual-uncased'    
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)     

max_seq_len = 512; batch_size = 12; 
epochs = 10; save_step=1 # Number of training epochs (authors recommend between 2 and 4)


# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    model_name, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 3, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)


optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


#================= Global variable ================= 

# # # If there's a GPU available...
if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
# device = torch.device("cpu")

# Copy the model to the GPU.
model.to(device)

#================= Functions ================= 
def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))
        

def choose_length_threshold(list_data):
    counter_tmp = collections.Counter(list_data).items() 
    list_threshold = []
    for i in range(0,max(list_data)+1):
        sum_val = 0
        for (key, val) in counter_tmp:
            if key<=i: sum_val+=val
         
        list_threshold.append((i, sum_val/len(list_data)*100.0))
    print(list_threshold) 

def create_data(data_file_path):
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0, names=['Content', 'NamedEntity', 'Polarity'])

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    Tweets = df.Content.values#[:5]
    Targets = df.NamedEntity.values#[:5]
    labels = df.Polarity.values#[:5]
    
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

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
# In[112]:

def write_csv_file_header(file_path, list_data, fieldnames):
    with open(file_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for line in list_data:
            writer.writerow(line)#({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})

def get_f1(results_folder, num_batch): 
    csv_writer = open_csv_file(results_folder + 'f1_avg.csv', ['num_batch', 'f1_avg'])
    for num_context_word in range(0, num_batch, 1):
        fname = "val_result_"+str(val_epoch)+".json"
        file_path = results_folder + fname
        dict_results = read_dict(file_path)
        f1_avg = round(dict_results["macro avg"]["f1-score"]*100,2)
        csv_writer.writerow([num_context_word, f1_avg])
#================= Functions ================= 

# from sklearn.model_selection import train_test_split

# # Use 90% for training and 10% for validation.
# train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,test_size=0.1)
# # Do the same for the masks.
# train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,test_size=0.1)

# train_type_inputs, validation_type_inputs, _, _ = train_test_split(token_type_ids, labels,test_size=0.1)


input_ids, labels, attention_masks, token_type_ids = create_data(data_file_path); 
train_inputs = torch.tensor(input_ids)
train_labels = torch.tensor(labels)
train_masks = torch.tensor(attention_masks)
train_type_inputs = torch.tensor(token_type_ids)

input_ids, labels, attention_masks, token_type_ids = create_data(data_file_path_dev)
validation_inputs = torch.tensor(input_ids)
validation_labels = torch.tensor(labels)
validation_masks = torch.tensor(attention_masks)
validation_type_inputs = torch.tensor(token_type_ids)

# The DataLoader needs to know our batch size for training, so we specify it 
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.



# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_type_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_type_inputs, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# from transformers import get_linear_schedule_with_warmup
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps)

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []; accuracy_values = []


fieldnames=['epoch','train_loss','val_acc', 'ifscore']
list_data = []
# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0; total_accuracy = 0   

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_type = batch[2].to(device)        
        b_labels = batch[3].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, 
                    token_type_ids=b_input_type, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # compute accuracy
        # Move logits and labels to CPU
#         logits = loss.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
        
        
        # Calculate the accuracy for this batch of test sentences.
#         tmp_train_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
#         total_accuracy += tmp_train_accuracy        
        
        
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
#         scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)   
#     avg_train_accuracy = total_accuracy/ len(train_dataloader)
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
#     accuracy_values.append(avg_train_accuracy)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
#     print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))    
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    
    pred_flat, labels_flat = [], []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_input_type, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=b_input_type, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

        pred_flat = pred_flat + list(np.argmax(logits, axis=1).flatten())
        labels_flat = labels_flat + list(label_ids.flatten())
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    class_report = classification_report(labels_flat, pred_flat, output_dict=True)

    
    #save model
    output_dir = data_folder +model_save + str(epoch_i)+"/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    log_path=data_folder +model_save+"log_model.csv"  
    val_acc = 1.0*eval_accuracy/nb_eval_steps
    ifscore = class_report["macro avg"]["f1-score"]
    if epoch_i%save_step==0:
        save_model(output_dir)
        list_data.append({'epoch':epoch_i,'train_loss':round(avg_train_loss,4), 'val_acc': round(val_acc,4), 'ifscore': round(ifscore,4)})
        print(list_data[-1])
        print(class_report)
        fname = "val_result_"+str(epoch_i)+".json"
        write_dict(class_report, data_folder_dev+fname)   
        
write_csv_file_header(log_path, list_data, fieldnames) 
print("Training complete!")



