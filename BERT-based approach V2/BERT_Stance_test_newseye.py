import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer, AutoTokenizer
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch, os, csv, codecs, json
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import collections
import argparse
from os import listdir
from os.path import isfile, join
import sys


# ================= Functions =================
def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))


def create_data(tokenizer, data_file_path):
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
                     names=['Content', 'LineNumber', 'NamedEntity', 'Polarity', 'LocalPosition',
                            'Offset'])

    print('Number of lines: {:,}\n'.format(df.shape[0]))

    Tweets = df.Content.values  # [:5]
    Targets = df.NamedEntity.values  # [:5]
    labels = df.Polarity.values  # [:5]
    lineNumbers = df.LineNumber.values
    file_names = np.array([data_file_path[data_file_path.rindex('/') + 1:].replace('.csv', '.txt')] * len(labels))

    print(collections.Counter(labels))

    input_ids = []

    for i in range(len(Tweets)):
        sent = Tweets[i]
        target_sent = Targets[i]

        encoded_sent = tokenizer.encode(
            sent, target_sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation_strategy='only_first',
            truncation=True,
            max_length=max_seq_len,  # Truncate all sentences.
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


    return input_ids, labels, attention_masks, token_type_ids, lineNumbers, file_names



parser = argparse.ArgumentParser()
parser.add_argument("curr_lang", help="Choose curr_lang", type=str)  # German, French, NLF, Swedish, PULS, EMM
parser.add_argument("cased", help="Choose cased", type=int)  # 1: cased
parser.add_argument("max_seq_len", help="Choose max_seq_len", type=int)
parser.add_argument("val_epoch", help="Choose val_epoch", type=int)  # 1 2 3

args = parser.parse_args()
curr_lang = args.curr_lang
flag_cased = args.cased
val_epoch = str(args.val_epoch)
max_seq_len = args.max_seq_len  # 273#data_ver3

if max_seq_len == 256:
    batch_size = 16
elif max_seq_len == 128:
    batch_size = 32
elif max_seq_len == 64:
    batch_size = 64

dataset_name = curr_lang + "_data_test_input"  # +"_"+str(max_seq_len)
data_folder = "../STANCE_DETECTION/" + dataset_name + "/"
data_file_path = data_folder + curr_lang + "_data_test.csv"


list_data_file_paths = [join(data_folder, f) for f in listdir(data_folder) if
                        isfile(join(data_folder, f)) and f.endswith('.csv')]


if flag_cased == 1:
    output_dir = "../STANCE_DETECTION/" + curr_lang + "_data_train_input" + "/model_save_cased_" + str(
        max_seq_len) + "/" + val_epoch
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(output_dir, do_lower_case=False)
    mark_cased = "cased"
else:
    output_dir = "../STANCE_DETECTION/" + curr_lang + "_data_train_input" + "/model_save_uncased_" + str(
        max_seq_len) + "/" + val_epoch
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(output_dir, do_lower_case=True)
    mark_cased = "uncased"


for i, file_path in enumerate(list_data_file_paths):
    input_ids_tmp, labels_tmp, attention_masks_tmp, token_type_ids_tmp, \
    lineNumbers_tmp, file_names_tmp = create_data(tokenizer, file_path)
    if i == 0:
        input_ids, labels, attention_masks, token_type_ids, lineNumbers, file_names = \
            input_ids_tmp, labels_tmp, attention_masks_tmp, token_type_ids_tmp, lineNumbers_tmp, file_names_tmp

    else:

        input_ids = np.append(input_ids, input_ids_tmp, axis=0)
        labels = np.append(labels, labels_tmp, axis=0)
        attention_masks = np.append(attention_masks, attention_masks_tmp, axis=0)
        token_type_ids = np.append(token_type_ids, token_type_ids_tmp, axis=0)
        lineNumbers = np.append(lineNumbers, lineNumbers_tmp, axis=0)
        file_names = np.append(file_names, file_names_tmp, axis=0)

print('Loading BERT model...')

if curr_lang == 'French':
    model = CamembertForSequenceClassification.from_pretrained(output_dir)
else:
    model = BertForSequenceClassification.from_pretrained(output_dir)


if torch.cuda.is_available():

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))


else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


model.to(device)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_type_inputs = torch.tensor(token_type_ids)
prediction_labels = torch.tensor(labels)


prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_type_inputs, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)



print('Predicting labels for {:,} test lines...'.format(len(prediction_inputs)))

model.eval()


predictions, true_labels = [], []


for batch in prediction_dataloader:

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_input_type, b_labels = batch


    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=b_input_type,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

flat_true_labels = [item for sublist in true_labels for item in sublist]

class_report = classification_report(flat_true_labels, flat_predictions, output_dict=True)

print(class_report)

result_folder = data_folder + 'results/'
if not os.path.exists(result_folder):
    os.mkdir(result_folder)

write_dict(class_report, result_folder + "test_result_" + mark_cased + "_" + val_epoch + ".json")

result_file_path = result_folder + curr_lang + "_data_test_result.json"

dict_result = {}
dict_stance = {'0': '+', '1': '-', '2': 'n'}
for i, j, k in zip(file_names, lineNumbers, flat_predictions):
    j = str(j)
    k = str(k)
    if i not in dict_result:
        dict_result[i] = {j: dict_stance[k]}
    else:
        dict_result[i][j] = dict_stance[k]

write_dict(dict_result, result_file_path)

print('Written ', result_file_path)

