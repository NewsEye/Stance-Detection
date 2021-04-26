#!/usr/bin/env python
# coding: utf-8



from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, AutoTokenizer
from transformers import CamembertForSequenceClassification, CamembertTokenizer
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


parser = argparse.ArgumentParser()
parser.add_argument("curr_lang", help="Choose curr_lang", type=str)  # Finnish, French, German, Swedish
parser.add_argument("cased", help="Choose cased", type=int)  # cased==1
parser.add_argument("max_seq_len", help="Choose max_seq_len", type=int)  # 128

args = parser.parse_args()
curr_lang = args.curr_lang;
flag_cased = args.cased
max_seq_len = args.max_seq_len  # 273#data_ver3



if max_seq_len == 256:
    batch_size = 16;
elif max_seq_len == 128:
    batch_size = 32
elif max_seq_len == 64:
    batch_size = 64

dataset_name = curr_lang + "_data_train_input"  # +"_"+str(max_seq_len)#German_data_dev_input_64
data_folder = "../STANCE_DETECTION/" + dataset_name + "/"
data_file_path = data_folder + curr_lang + "_data_train.csv"

dataset_name_dev = curr_lang + "_data_dev_input"  # +"_"+str(max_seq_len)
data_folder_dev = "../STANCE_DETECTION/" + dataset_name_dev + "/"
data_file_path_dev = data_folder_dev + curr_lang + "_data_dev.csv"

data_folder_model = "../STANCE_DETECTION/" + curr_lang + "_data_train_input/"


if flag_cased == 1:
    model_save = "model_save_cased_" + str(max_seq_len) + "/"
    if curr_lang == 'German':
        model_name = data_folder_model + 'bert-base-german-europeana-cased'  # 'bert-base-german-dbmdz-cased'
    elif curr_lang == 'Finnish':
        model_name = data_folder_model + 'bert-base-finnish-cased-v1'  # 'bert-base-finnish-cased-v1'#data_folder_model+'bert-base-finnish-cased-v1'
    elif curr_lang == 'Swedish':
        model_name = data_folder_model + 'bert-base-swedish-cased'  # 'bert-base-multilingual-cased'  # 'bert-base-swedish-cased'
    elif curr_lang == 'EMM' or curr_lang == 'PULS':
        model_name = 'bert-base-cased'
    elif curr_lang == 'French':
        model_name = data_folder_model + 'camembert-base'

    if curr_lang == 'French':
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
else:
    model_save = "model_save_uncased_" + str(max_seq_len) + "/"
    if curr_lang == 'German':
        model_name = data_folder_model + 'bert-base-german-europeana-uncased'  # 'bert-base-german-dbmdz-cased'
    elif curr_lang == 'Finnish':
        model_name = data_folder_model + 'bert-base-finnish-uncased-v1'  # 'bert-base-finnish-cased-v1'#data_folder_model+'bert-base-finnish-cased-v1'
    elif curr_lang == 'Swedish':
        model_name = data_folder_model + 'bert-base-swedish-uncased'  # 'bert-base-multilingual-cased'  # 'bert-base-swedish-cased'
    elif curr_lang == 'EMM' or curr_lang == 'PULS':
        model_name = 'bert-base-uncased'
    elif curr_lang == 'French':
        model_name = data_folder_model + 'camembert-base'

    if curr_lang == 'French':
        tokenizer = CamembertTokenizer.from_pretrained(model_name, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

epochs = 10
save_step = 1  
if curr_lang == 'French':
    model = CamembertForSequenceClassification.from_pretrained(
        model_name,  
        num_labels=3,  
        output_attentions=False,  
        output_hidden_states=False,  
    )
else:
    model = BertForSequenceClassification.from_pretrained(
        model_name,  
        num_labels=3,  
        output_attentions=False,  
        output_hidden_states=False,  
    )

optimizer = AdamW(model.parameters(),
                  lr=5e-5,  
                  eps=1e-8  
                  )

# ================= Global variable =================


if torch.cuda.is_available():

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))


else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


model.to(device)


# ================= Functions =================
def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))


def read_dict(dict_file_path):
    with codecs.open(dict_file_path, encoding='utf-8') as myfile:
        dict_tmp = json.load(myfile)
    return dict_tmp


def choose_length_threshold(list_data):
    counter_tmp = collections.Counter(list_data).items()
    list_threshold = []
    for i in range(0, max(list_data) + 1):
        sum_val = 0
        for (key, val) in counter_tmp:
            if key <= i: sum_val += val

        list_threshold.append((i, sum_val / len(list_data) * 100.0))
    print(list_threshold)


def create_data(data_file_path):
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
                     names=['Content', 'LineNumber', 'NamedEntity', 'Polarity', 'LocalPosition',
                            'Offset'])  # Content,LineNumber,NamedEntity,Polarity
    df = df.sample(frac=1)


    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    Tweets = df.Content.values  # [:5]
    Targets = df.NamedEntity.values  # [:5]
    labels = df.Polarity.values  # [:5]
    lineNumbers = df.LineNumber.values

    print(collections.Counter(labels))


    input_ids = []


    for i in range(len(Tweets)):
        sent = Tweets[i]
        target_sent = Targets[i]


        encoded_sent = tokenizer.encode(
            sent, target_sent,  
            add_special_tokens=True,  
            truncation_strategy='only_first',
            truncation=True,

            max_length=max_seq_len,  
        )


        input_ids.append(encoded_sent)


    token_type_ids = []

    attention_masks = []

    for sent in input_ids:

        start_second_sent_index = sent.index(tokenizer.sep_token_id) + 1
        token_type_id = [0 if i < start_second_sent_index else 1 for i in range(len(sent))]
        if curr_lang == 'French':
            token_type_id = [0] * len(sent)
        token_type_ids.append(token_type_id)

        att_mask = [int(token_id > 0) for token_id in sent]


        attention_masks.append(att_mask)

    input_ids = pad_sequences(input_ids, maxlen=max_seq_len, dtype="long",
                              value=0, truncating="post", padding="post")
    token_type_ids = pad_sequences(token_type_ids, maxlen=max_seq_len, dtype="long",
                                   value=0, truncating="post", padding="post")
    attention_masks = pad_sequences(attention_masks, maxlen=max_seq_len, dtype="long",
                                    value=0, truncating="post", padding="post")

    print('input_ids[0] AFTER', input_ids[0])
    print('token_type_ids AFTER', token_type_ids[0])
    print('mask_ids[0] AFTER', attention_masks[0])

    return input_ids, labels, attention_masks, token_type_ids




def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



def write_csv_file_header(file_path, list_data, fieldnames):
    with open(file_path, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for line in list_data:
            writer.writerow(line)  

def get_f1(results_folder, num_batch):
    csv_writer = open_csv_file(results_folder + 'f1_avg.csv', ['num_batch', 'f1_avg'])
    for num_context_word in range(0, num_batch, 1):
        fname = "val_result_" + str(val_epoch) + ".json"
        file_path = results_folder + fname
        dict_results = read_dict(file_path)
        f1_avg = round(dict_results["macro avg"]["f1-score"] * 100, 2)
        csv_writer.writerow([num_context_word, f1_avg])



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


# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_type_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_type_inputs, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs


# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = [];
accuracy_values = []

fieldnames = ['epoch', 'train_loss', 'val_acc', 'ifscore']
list_data = []
# For each epoch...
for epoch_i in range(0, epochs):


    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')


    t0 = time.time()


    total_loss = 0
    total_accuracy = 0

    model.train()


    for step, batch in enumerate(train_dataloader):


        if step % 50 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)


            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_type = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=b_input_type,
                        attention_mask=b_input_mask,
                        labels=b_labels)


        loss = outputs[0]

        total_loss += loss.item()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


        optimizer.step()



    avg_train_loss = total_loss / len(train_dataloader)


    loss_values.append(avg_train_loss)


    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================


    print("")
    print("Running Validation...")

    t0 = time.time()


    model.eval()


    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    pred_flat, labels_flat = [], []


    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)


        b_input_ids, b_input_mask, b_input_type, b_labels = batch


        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=b_input_type,
                            attention_mask=b_input_mask)

        logits = outputs[0]


        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()


        tmp_eval_accuracy = flat_accuracy(logits, label_ids)


        eval_accuracy += tmp_eval_accuracy


        nb_eval_steps += 1

        pred_flat = pred_flat + list(np.argmax(logits, axis=1).flatten())
        labels_flat = labels_flat + list(label_ids.flatten())
        

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    class_report = classification_report(labels_flat, pred_flat, output_dict=True)


    output_dir = data_folder + model_save + str(epoch_i) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_path = data_folder + model_save + "log_model.csv"
    val_acc = 1.0 * eval_accuracy / nb_eval_steps
    ifscore = class_report["macro avg"]["f1-score"]
    if epoch_i % save_step == 0:
        save_model(output_dir)
        list_data.append({'epoch': epoch_i, 'train_loss': round(avg_train_loss, 4), 'val_acc': round(val_acc, 4),
                          'ifscore': round(ifscore, 4)})
        print(list_data[-1])
        print(class_report)
        fname = str(flag_cased) + "_val_result_" + str(epoch_i) + ".json"
        write_dict(class_report, data_folder_dev + fname)

write_csv_file_header(log_path, list_data, fieldnames)
print("Training complete!")
