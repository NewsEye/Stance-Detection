#!/usr/bin/env python
# coding: utf-8
from nltk.corpus import sentiwordnet as swn
import codecs, re
import os, string, json
from stop_words import get_stop_words
import sklearn.metrics as metrics
import unicodedata, collections
import csv
import pandas as pd
import argparse
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import itertools
import numpy as np


# ================List of functions================

def open_csv_file(file_name, list_headers):
    csv_data = open(file_name, mode='w')
    csv_writer = csv.writer(csv_data, delimiter=',', quotechar='"')
    csv_writer.writerow(list_headers)  # ['Content', 'NamedEntity', 'Polarity']
    return csv_writer, csv_data


def convert_file_dicts_all(input_folder, file_name):
    dict_toks = {}
    dict_NEs = {}
    dict_global_local_index = {}
    dict_lines_words = {}

    if file_name not in dict_toks:
        dict_toks[file_name] = []
        dict_NEs[file_name] = []
        dict_lines_words[file_name] = []
        dict_global_local_index[file_name] = {}
    file_path = input_folder + '/' + file_name
    text = codecs.open(file_path, 'rb', encoding="utf-8").read().split("\n")
    tok_index = 1
    tmp_NE = ""
    tmp_NE_index = -1
    tmp_NE_offset = 1
    pol = ""
    count_line = 1
    for line in text:

        list_line_tokens = []
        if line.strip() == '':
            #             print("######");
            count_line += 1
            tok_index += 1
            dict_lines_words[file_name].append(list_line_tokens)
            #             if tmp_NE != "":
            #                 tmp_NE_upper = tmp_NE.strip().upper()
            #                 dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset), tmp_NE_upper))
            continue

        lastTokenPos = 0
        local_tok_index = 0
        line = line.rstrip('SpaceAfter').strip()
        for spacePos in re.finditer(r"$|\n", line):
            tokenEndPos = spacePos.start()
            tokenNE = line[lastTokenPos:tokenEndPos]
            list_line_tokens.append(tokenNE)

            # print(file_name, tokenNE)
            token = tokenNE.split()[0]
            tag_NE = tokenNE.split()[1]

            if len(tag_NE) > 0:

                if tag_NE[0] == 'B' or tag_NE[0] == 'S':

                    if tmp_NE != "" and pol != '':
                        tmp_NE_upper = tmp_NE.strip()  # .upper()
                        dict_NEs[file_name].append(
                            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper, ne, nel))
                    ne = tokenNE.split()[1]
                    pol = tokenNE.split()[-1]
                    nel = tokenNE.split()[-2]
                    tmp_NE = token + " ";
                    tmp_NE_offset = 1;
                    tmp_NE_index = tok_index

                elif tag_NE[0] == 'O':
                    if tmp_NE != "" and pol != '':
                        tmp_NE_upper = tmp_NE.strip()  # .upper();
                        dict_NEs[file_name].append(
                            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper, ne, nel))

                    tmp_NE = "";
                    tmp_NE_offset = 1;
                    tmp_NE_index = -1

                elif (tag_NE[0] == 'I' or tag_NE[0] == 'E') and tmp_NE != '':
                    tmp_NE += token + " "
                    tmp_NE_offset += 1

            lastTokenPos = tokenEndPos + 1
            dict_toks[file_name].append((tok_index, token))
            dict_global_local_index[file_name][tok_index] = (count_line, local_tok_index)

            local_tok_index += 1;  # tok_index += 1

        count_line += 1
        tok_index += 1
        dict_lines_words[file_name].append(list_line_tokens)

    if tmp_NE != "":
        tmp_NE_upper = tmp_NE.strip()  # .upper()
        dict_NEs[file_name].append(
            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper, ne, nel))

    return dict_toks, dict_NEs, dict_global_local_index, dict_lines_words


def convert_file_dicts_2020(input_folder, file_name):
    dict_toks = {}
    dict_NEs = {}
    dict_global_local_index = {}
    dict_lines_words = {}

    if file_name not in dict_toks:
        dict_toks[file_name] = []
        dict_NEs[file_name] = []
        dict_lines_words[file_name] = []
        dict_global_local_index[file_name] = {}
    file_path = input_folder + '/' + file_name
    text = codecs.open(file_path, 'rb', encoding="utf-8").read().split("\n")
    tok_index = 1
    tmp_NE = ""
    tmp_NE_index = -1
    tmp_NE_offset = 1
    pol = ""
    count_line = 1
    for line in text:

        list_line_tokens = []
        if line.strip() == '':
            #             print("######");
            count_line += 1
            tok_index += 1
            dict_lines_words[file_name].append(list_line_tokens)
            dict_toks[file_name].append((tok_index, ''))
            continue

        lastTokenPos = 0
        local_tok_index = 0
        line = line.rstrip('SpaceAfter').strip()
        for spacePos in re.finditer(r"$|\n", line):
            tokenEndPos = spacePos.start()
            tokenNE = line[lastTokenPos:tokenEndPos]
            list_line_tokens.append(tokenNE)

            print(file_name, tokenNE, count_line)
            token = tokenNE.split()[0]
            tag_NE = tokenNE.split()[1]

            if len(tag_NE) > 0:

                if tag_NE[0] == 'B' or tag_NE[0] == 'S':

                    if tmp_NE != "" and pol != '':
                        tmp_NE_upper = tmp_NE.strip()  # .upper()
                        dict_NEs[file_name].append(
                            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper))

                    pol = tokenNE.split()[-1]
                    # assert pol in ['+', '-', 'n']
                    tmp_NE = token + " "
                    tmp_NE_offset = 1
                    tmp_NE_index = tok_index

                elif tag_NE[0] == 'O':
                    if tmp_NE != "" and pol != '':
                        tmp_NE_upper = tmp_NE.strip()  # .upper();
                        dict_NEs[file_name].append(
                            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper))

                    tmp_NE = ""
                    tmp_NE_offset = 1
                    tmp_NE_index = -1

                elif (tag_NE[0] == 'I' or tag_NE[0] == 'E') and tmp_NE != '':
                    tmp_NE += token + " "
                    tmp_NE_offset += 1

            lastTokenPos = tokenEndPos + 1
            dict_toks[file_name].append((tok_index, token))
            dict_global_local_index[file_name][tok_index] = (count_line, local_tok_index)

            local_tok_index += 1  # tok_index += 1

        count_line += 1
        tok_index += 1
        dict_lines_words[file_name].append(list_line_tokens)
        # dict_toks[file_name].append((tok_index, ''))

    if tmp_NE != "":
        tmp_NE_upper = tmp_NE.strip()  # .upper()
        dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper))

    return dict_toks, dict_NEs, dict_global_local_index, dict_lines_words


def convert_file_dicts_3(input_folder, file_name):
    dict_toks = {}
    dict_NEs = {}
    dict_global_local_index = {}
    dict_lines_words = {}

    if file_name not in dict_toks:
        dict_toks[file_name] = []
        dict_NEs[file_name] = []
        dict_lines_words[file_name] = []
        dict_global_local_index[file_name] = {}
    file_path = input_folder + '/' + file_name
    text = codecs.open(file_path, 'rb', encoding="utf-8").read().split("\n")
    tok_index = 0
    tmp_NE = ""
    tmp_NE_index = 0
    tmp_NE_offset = 1
    pol = ""
    count_line = 0
    for line in text:

        list_line_tokens = []
        if line.strip() == '' or line.startswith("# --"):
            #             print("######");
            count_line += 1
            tok_index += 1
            dict_lines_words[file_name].append(list_line_tokens)
            dict_toks[file_name].append((tok_index, ''))
            continue

        lastTokenPos = 0
        local_tok_index = 0
        line = line.strip().rstrip('SpaceAfter').strip()
        for spacePos in re.finditer(r"$|\n", line):
            tokenEndPos = spacePos.start()
            tokenNE = line[lastTokenPos:tokenEndPos]
            list_line_tokens.append(tokenNE)

            print(file_name, tokenNE, count_line)
            token = tokenNE.split()[0]
            tag_NE = tokenNE.split()[1]

            if len(tag_NE) > 0:

                if tag_NE[0] == 'B' or tag_NE[0] == 'S':

                    if tmp_NE != "" and pol != '':
                        tmp_NE_upper = tmp_NE.strip()  # .upper()
                        dict_NEs[file_name].append(
                            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper))

                    pol = tokenNE.split()[-1]
                    # assert pol in ['+', '-', 'n']
                    tmp_NE = token + " "
                    tmp_NE_offset = 1
                    tmp_NE_index = tok_index

                elif tag_NE[0] == 'O':
                    if tmp_NE != "" and pol != '':
                        tmp_NE_upper = tmp_NE.strip()  # .upper();
                        dict_NEs[file_name].append(
                            (str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper))

                    tmp_NE = ""
                    tmp_NE_offset = 1
                    tmp_NE_index = -1

                elif (tag_NE[0] == 'I' or tag_NE[0] == 'E') and tmp_NE != '':
                    tmp_NE += token + " "
                    tmp_NE_offset += 1

            lastTokenPos = tokenEndPos + 1
            dict_toks[file_name].append((tok_index, token))
            dict_global_local_index[file_name][tok_index] = (count_line, local_tok_index)
            local_tok_index += 1  # tok_index += 1

        count_line += 1
        tok_index += 1
        dict_lines_words[file_name].append(list_line_tokens)
        # dict_toks[file_name].append((tok_index, ''))

    if tmp_NE != "":
        tmp_NE_upper = tmp_NE.strip()  # .upper()
        dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset) + ":" + pol, tmp_NE_upper))

    return dict_toks, dict_NEs, dict_global_local_index, dict_lines_words


def create_NEs_csv(file_name, dict_NEs, dict_toks, dict_NEs_senti, csv_writer, lang=""):
    # Content,NamedEntity,Polarity
    max_seq_len = 128
    if file_name not in dict_NEs_senti:
        dict_NEs_senti[file_name] = []

    for ne_item in dict_NEs[file_name]:
        ne_pos = int(ne_item[0].split(":")[0])
        ne_offset = int(ne_item[0].split(":")[1])

        # ne_pol = ne_item[0].split(":")[2].strip()
        # ne_pol = int(ne_pol.replace('+', '0').replace('-', '1').replace('n', '2'))

        ne_val = ne_item[1]
        content = dict_toks[file_name][
                  ne_pos + ne_offset - int(max_seq_len / 2):ne_pos + ne_offset + int(max_seq_len / 2)]
        content = ' '.join([word for (_, word) in content])
        if len(content.strip()) == 0:
            content = ne_val
        csv_writer.writerow([content, ne_pos, ne_val, ne_offset])

    return dict_NEs_senti


# connection = sqlite3.connect('../data/database.db')

def create_NEs_sql(file_name, dict_NEs, connection):
    cur = connection.cursor()
    for ne_id, ne_item in enumerate(dict_NEs[file_name]):
        ne_pos = int(ne_item[0].split(':')[0])
        ne_offset = int(ne_item[0].split(':')[1])
        ne_stance = ne_item[0].split(":")[2]
        ne_text = ne_item[1]
        ne_type = ne_item[2]
        ne_nel = ne_item[3]

        file_name_new = file_name.replace('.txt', '')
        print(file_name)
        print(file_name_new)
        cur.execute("INSERT INTO NamedEntity (text,type,stance,important_words,ne_position,num_toks,nel,file_name) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ne_text, ne_type, ne_stance, ' ', ne_pos, ne_offset, ne_nel, file_name_new))
    cur.close()


def iob_2_sql(input_folder, file_name, connection):
    dict_toks, dict_NEs, _, _ = convert_file_dicts_all(input_folder, file_name)
    create_NEs_sql(file_name, dict_NEs, connection)


def create_file_csv_2(input_folder, file_name, csv_writer):
    dict_toks, dict_NEs, dict_global_local_index, dict_lines_words = convert_file_dicts_3(input_folder, file_name)

    dict_NEs_senti = {}
    dict_NEs_senti = create_NEs_csv(file_name, dict_NEs, dict_toks, dict_NEs_senti, csv_writer)


def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))


def read_dict(dict_file_path):
    with codecs.open(dict_file_path, encoding='utf-8') as myfile:
        dict_tmp = json.load(myfile)
    return dict_tmp


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def create_NEs_csv_train_dev(file_name, dict_NEs, dict_toks, dict_NEs_senti, csv_writer, set_lines, split_data=''):
    # Content,NamedEntity,Polarity
    max_seq_len = 128
    if file_name not in dict_NEs_senti:
        dict_NEs_senti[file_name] = []

    for ne_item in dict_NEs[file_name]:

        ne_pos = int(ne_item[0].split(":")[0])
        if ne_pos not in set_lines:
            continue

        ne_offset = int(ne_item[0].split(":")[1])

        ne_pol = ne_item[0].split(":")[2].strip()
        ne_pol = int(ne_pol.replace('+', '0').replace('-', '1').replace('n', '2'))

        ne_val = ne_item[1]
        ne_local_pos = max(ne_pos + ne_offset - int(max_seq_len / 2), 0)
        content = dict_toks[file_name][
                  ne_local_pos:ne_pos + ne_offset + int(max_seq_len / 2)]
        ne_local_pos = ne_pos - ne_local_pos - 1
        content = ' '.join([word for (_, word) in content])
        if len(content.strip()) == 0:
            content = ne_val
        csv_writer.writerow([content, ne_pos, ne_val, ne_pol, ne_local_pos, ne_offset])
        if split_data == 'train' and ne_pol == 0 or ne_pol == 1:  # only for augment train data
            csv_writer.writerow([content, ne_pos, ne_val, ne_pol, ne_local_pos, ne_offset])

    return dict_NEs_senti


def create_file_csv_train_dev(input_folder, file_name, csv_writer, set_lines, output_file, split_data):
    dict_toks, dict_NEs, dict_global_local_index, dict_lines_words = convert_file_dicts_3(input_folder, file_name)

    dict_NEs_senti = {}
    dict_NEs_senti = create_NEs_csv_train_dev(file_name, dict_NEs, dict_toks, dict_NEs_senti, csv_writer,
                                              set_lines, split_data)


def txt2csv_train_dev(input_folder, input_file, output_file, split_data):
    input_data_folder = input_folder + "data/"
    for file_name in os.listdir(input_data_folder):
        if '.ipynb_checkpoints' in file_name or '.txt' not in file_name:
            continue
        csv_writer, csv_data = open_csv_file(input_data_folder + '/' + output_file,
                                             ['Content', 'LineNumber', 'NamedEntity',
                                              'Polarity', 'LocalPosition', 'Offset'])

        df = pd.read_csv(input_data_folder + input_file,
                         names=['Content', 'LineNumber', 'NamedEntity', 'Polarity'],
                         delimiter=',', lineterminator='\n', encoding="utf-8", header=0)

        list_lines = df['LineNumber'].tolist()
        # print([item for item, count in collections.Counter(set_lines).items() if count > 1])
        #
        # print(len(set_lines));exit()
        # content, ne_pos, ne_val, ne_pol, ne_local_pos, ne_offset
        print(file_name)
        create_file_csv_train_dev(input_data_folder, file_name, csv_writer, list_lines, output_file, split_data)
        csv_data.close()


def get_content_from_ne_pos(file_path, ne_pos, ne_offset):
    # file_name = i_pos.split(':')[0]
    # ne_pos = int(i_pos.split(':')[1])
    # ne_offset = int(i_pos.split(':')[2])
    max_length = 128
    orig_texts = open(file_path).read().split("\n")

    texts = orig_texts[max([0, ne_pos - int(max_length / 2)]):ne_pos + ne_offset + int(max_length / 2)]
    texts = [word.split()[0] if len(word.split()) > 0 else '\n' for word in texts]

    if (ne_pos - int(max_length / 2)) > 0 and (ne_pos + ne_offset + int(max_length / 2) <= len(texts)):
        assert len(texts) == 128 + ne_offset, 'len(texts), 128+ne_offset' % (len(texts), 128 + ne_offset)
    return ' '.join(texts)


def create_NEs_csv_train_dev_beg(input_folder, file_name, dict_NEs, dict_toks, dict_NEs_senti, csv_writer):
    # Content,NamedEntity,Polarity
    max_seq_len = 128
    if file_name not in dict_NEs_senti:
        dict_NEs_senti[file_name] = []

    for ne_item in dict_NEs[file_name]:
        ne_pos = int(ne_item[0].split(":")[0])
        ne_offset = int(ne_item[0].split(":")[1])
        ne_pol = ne_item[0].split(":")[2].strip()
        print(ne_item)
        ne_pol = int(ne_pol.replace('+', '0').replace('-', '1').replace('n', '2').replace('_', '3'))

        ne_val = ne_item[1]
        start_pos = max(ne_pos - int(max_seq_len / 2), 0)
        ne_local_pos = ne_pos - start_pos  # 2
        content = dict_toks[file_name][
                  start_pos:ne_pos + ne_offset + int(max_seq_len / 2)]
        check_local_pos = ' '.join([word for (_, word) in content[ne_local_pos:ne_local_pos + ne_offset]])
        print('nel_val, local_val: {0}; {1}; '.format(ne_val, check_local_pos))
        assert ne_val == check_local_pos, (ne_val, check_local_pos)
        content = ' '.join([word for (_, word) in content])

        csv_writer.writerow([content, ne_pos, ne_val, ne_pol, ne_local_pos, ne_offset])

    return dict_NEs_senti


def create_file_csv_train_dev_beg(input_folder, file_name, csv_writer):
    dict_toks, dict_NEs, dict_global_local_index, dict_lines_words = convert_file_dicts_3(input_folder, file_name)

    dict_NEs_senti = {}
    dict_NEs_senti = create_NEs_csv_train_dev_beg(input_folder, file_name, dict_NEs, dict_toks, dict_NEs_senti,
                                                  csv_writer)


def txt2csv_train_dev_beg(input_folder, output_file):
    input_data_folder = input_folder + "data/"
    csv_writer, csv_data = open_csv_file(output_file,
                                         ['Content', 'LineNumber', 'NamedEntity',
                                          'Polarity', 'LocalPosition', 'Offset',
                                          'Year', 'FileName'])
    for file_name in os.listdir(input_data_folder):
        print(file_name)
        if '.ipynb_checkpoints' in file_name or '.txt' not in file_name:
            continue
        create_file_csv_train_dev_beg(input_data_folder, file_name, csv_writer)

    csv_data.close()


def txt2csv(input_folder):
    input_data_folder = input_folder + "data/"
    out_data_folder = input_folder + "data_csv/"

    if not os.path.exists(out_data_folder):
        os.mkdir(out_data_folder)

    for file_name in os.listdir(input_data_folder):
        if '.ipynb_checkpoints' in file_name or '.txt' not in file_name:
            continue
        print(file_name)
        csv_writer, csv_data = open_csv_file(out_data_folder + '/' + file_name.replace('.txt', '.csv'),
                                             ['Content', 'LineNumber', 'NamedEntity',
                                              'Polarity', 'LocalPosition', 'Offset'])
        create_file_csv_train_dev_beg(input_data_folder, file_name, csv_writer)
        csv_data.close()

def txt2csv_2021(input_folder):
    input_data_folder = out_data_folder = input_folder

    if not os.path.exists(out_data_folder):
        os.mkdir(out_data_folder)

    for file_name in os.listdir(input_data_folder):
        if '.ipynb_checkpoints' in file_name or '.txt' not in file_name:
            continue
        print(file_name)
        csv_writer, csv_data = open_csv_file(out_data_folder + '/' + file_name.replace('.txt', '.csv'),
                                             ['Content', 'LineNumber', 'NamedEntity',
                                              'Polarity', 'LocalPosition', 'Offset'])
        create_file_csv_train_dev_beg(input_data_folder, file_name, csv_writer)
        csv_data.close()


def write_txt_output(input_folder, flag_cased, curr_lang=''):
    input_data_folder = input_folder + "data/"
    if flag_cased == 1:
        result_folder = ''.join([input_folder, 'results_cased_3/'])
        output_folder = input_folder + 'output_cased_3/'
    else:
        result_folder = ''.join([input_folder, 'results_uncased_3/'])
        output_folder = input_folder + 'output_uncased_3/'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    for rs_file_name in os.listdir(result_folder):  #
        if '.json' not in rs_file_name:
            continue
        print(rs_file_name)
        result_file_path = result_folder + rs_file_name
        dict_results = read_dict(result_file_path)
        for key in dict_results:
            file_name = key
            file_path = input_data_folder + file_name
            output_file_path = output_folder + file_name
            lines = open(file_path, 'r', encoding='utf-8').read().split('\n')
            with open(output_file_path, 'w', encoding='utf-8') as fwrite:
                for i, line in enumerate(lines):
                    if str(i) not in dict_results[file_name]:
                        # if str(i + 1) not in dict_results[file_name]:
                        if i + 1 == len(lines):
                            fwrite.write(line)
                        else:
                            fwrite.write(line + '\n')
                    else:
                        toks = line.split()
                        # new_lines = '\t'.join(
                        #     toks[:len(toks) - 1] + [dict_results[file_name][str(i + 1)]] + toks[
                        #                                                                    len(toks) - 1:]).strip()
                        if curr_lang.lower() == 'french':  # only French uses SD data due to missing data
                            toks[-2] = dict_results[file_name][str(i + 1)]

                        else:
                            new_lines = '\t'.join(
                                toks[:len(toks) - 1] + [dict_results[file_name][str(i)]] + toks[
                                                                                           len(toks) - 1:]).strip()

                        if i + 1 == len(lines):
                            fwrite.write(new_lines)
                        else:
                            fwrite.write(new_lines + '\n')

            print('Finish writing txt output file ', output_file_path)


def write_txt_output_orig(input_folder, flag_cased):
    input_data_folder = input_folder + "data/"
    if flag_cased == 1:
        result_file_path = ''.join([input_folder, 'results_cased/', curr_lang, "_data_test_result.json"])
        output_folder = input_folder + 'output_cased/'
    else:
        result_file_path = ''.join([input_folder, 'results_uncased/', curr_lang, "_data_test_result.json"])
        output_folder = input_folder + 'output_uncased/'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    dict_results = read_dict(result_file_path)
    for file_name in os.listdir(input_data_folder):
        if file_name not in dict_results and ('.ipynb_checkpoints' in file_name or '.txt' not in file_name):
            continue
        print(file_name)
        file_path = input_data_folder + file_name
        output_file_path = output_folder + file_name
        lines = open(file_path, 'r', encoding='utf-8').read().split('\n')
        with open(output_file_path, 'w', encoding='utf-8') as fwrite:
            for i, line in enumerate(lines):
                if str(i + 1) not in dict_results[file_name]:
                    fwrite.write(line + '\n')
                else:
                    toks = line.split()
                    new_lines = '\t'.join(
                        toks[:len(toks) - 1] + [dict_results[file_name][str(i + 1)]] + toks[len(toks) - 1:]).strip()
                    fwrite.write(new_lines + '\n')

        print('Finish writing txt output file ', output_file_path)


def combine_nel_stance(nel_folder, stance_folder, result_folder):
    list_exception = []
    with open('list_exception.txt', 'w') as f_exception:
        #         id_ex = listdir(stance_folder).index('uusi_aura_1190907.txt')
        #         print(id_exs); exit()
        for file_name in os.listdir(stance_folder):  # [listdir(stance_folder)[id_ex]]:
            #             print(file_name)
            nel_file_path = nel_folder + file_name.replace('.txt', '-mel+ext.tsv')
            with open(nel_file_path, encoding='utf-8') as nel_file:
                nel_lines = nel_file.read().split('\n')
                stance_file_path = stance_folder + file_name
                with open(stance_file_path, encoding='utf-8') as stance_file:
                    # new_name = file_name.replace('.txt', '-mel+ext.tsv')
                    stance_lines = stance_file.read().split('\n')
                    if len(nel_lines) != len(stance_lines):
                        list_exception.append(stance_file_path)
                        f_exception.write(stance_file_path + "\n")
                        print('EX: ', file_name)
                        continue
                    result_file_path = result_folder + file_name
                    with open(result_file_path, 'w', encoding='utf-8') as result_file:
                        #                         print(file_name)
                        for i in range(len(nel_lines)):
                            #                             if len(nel_lines[i].strip())==0:
                            #                                 print("EX ", i, nel_lines[i])
                            #                                 result_file.write("\n")
                            #                                 continue
                            #                             print('line ',i)
                            #                             print('nel_line ', nel_lines[i])
                            #                             print('stance_line ', stance_lines[i])
                            nel_line = nel_lines[i].split()
                            stance_line = stance_lines[i].split()
                            if i == 0:
                                result_line = stance_line[:-1] + ['SD'] + stance_line[-1:]
                            else:
                                result_line = nel_line + stance_line[len(nel_line):]

                            if i + 1 == len(nel_lines):
                                result_file.write('\t'.join(result_line))
                            else:
                                result_file.write('\t'.join(result_line) + '\n')


def combine_nel_empty_stance(nel_folder, stance_folder, result_folder):
    with open('list_exception_2.txt', 'w') as f_exception:
        for file_name in os.listdir(nel_folder):
            nel_file_path = nel_folder + file_name

            with open(nel_file_path, encoding='utf-8') as nel_file:

                nel_lines = nel_file.read().split('\n')

                stance_file_path = stance_folder + file_name.replace('-mel+ext.tsv', '.txt')
                #                 print(stance_file_path)
                if not os.path.exists(stance_file_path):
                    result_file_path = result_folder + file_name.replace('-mel+ext.tsv', '.txt')
                    with open(result_file_path, 'w', encoding='utf-8') as result_file:
                        for i in range(len(nel_lines)):
                            if nel_lines[i].strip() == "":
                                if i + 1 < len(nel_lines):
                                    result_file.write("\n")
                                continue
                            nel_line = nel_lines[i].split()

                            if i == 0:
                                result_line = nel_line[:-2] + ['NEL-LIT', 'NEL-METO', 'SD', 'MISC']
                            else:
                                result_line = nel_line + ['_'] + ['_']

                            if i + 1 == len(nel_lines):
                                result_file.write('\t'.join(result_line))
                            else:
                                result_file.write('\t'.join(result_line) + '\n')

                    f_exception.write('NO FILE: ' + nel_file_path + "\n")
                    print('EXEPTION NO FILE: ', nel_file_path)
                    continue

                with open(stance_file_path, encoding='utf-8') as stance_file:
                    # new_name = file_name.replace('.txt', '-mel+ext.tsv')
                    stance_lines = stance_file.read().split('\n')
                    if len(stance_lines) != len(nel_lines):
                        f_exception.write('NO LINE: ', stance_file_path + "\n")
                        print('EXEPTION NO LINE: ', stance_file_path)
                        continue

                    result_file_path = result_folder + file_name.replace('-mel+ext.tsv', '.txt')
                    with open(result_file_path, 'w', encoding='utf-8') as result_file:
                        #                         print(file_name)
                        for i in range(len(stance_lines)):
                            stance_line = stance_lines[i].split()
                            nel_line = nel_lines[i].split()
                            if i == 0:
                                result_line = stance_line[:-1] + ['SD'] + stance_line[-1:]
                            else:
                                result_line = nel_line + stance_line[len(nel_line):]

                            if i + 1 == len(stance_lines):
                                result_file.write('\t'.join(result_line))
                            else:
                                result_file.write('\t'.join(result_line) + '\n')


def divide_data(data_file_path, data_train_path, data_dev_path, data_test_path):
    columns_names = ['Content', 'LineNumber', 'NamedEntity',
                     'Polarity', 'LocalPosition', 'Offset',
                     'Year', 'FileName']
    # data_file_path = input_folder + dataset + ".csv"
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
                     names=columns_names)

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    contents = df.Content.values
    lineNumbers = df.LineNumber.values
    namedEntities = df.NamedEntity.values
    polarities = df.Polarity.values
    localPositions = df.LocalPosition.values
    offsets = df.Offset.values
    years = df.Year.values
    fileNames = df.FileName.values
    print('set(years) ', set(years))
    print(collections.Counter(df.Polarity.values))
    dict_indices = {}  # neu, pos, neg
    for i_year in set(years):
        dict_indices[i_year] = {}

    for i in range(len(polarities)):
        polar = polarities[i]
        year = years[i]
        if polar == 2:  # 'n':
            if 'neu' not in dict_indices[year]:
                dict_indices[year]['neu'] = [i]
            else:
                dict_indices[year]['neu'].append(i)
        elif polar == 0:  # '+':
            if 'pos' not in dict_indices[year]:
                dict_indices[year]['pos'] = [i]
            else:
                dict_indices[year]['pos'].append(i)
        elif polar == 1:  # '-':
            if 'neg' not in dict_indices[year]:
                dict_indices[year]['neg'] = [i]
            else:
                dict_indices[year]['neg'].append(i)

    df_trains, df_devs, df_tests = [], [], []
    for i_year in set(years):
        print(i_year, len(dict_indices[i_year]['pos']), len(dict_indices[i_year]['neg']),
              len(dict_indices[i_year]['neu']))

        data_pos = {'Content': contents[dict_indices[i_year]['pos']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['pos']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['pos']],
                    'Polarity': polarities[dict_indices[i_year]['pos']],
                    'LocalPosition': localPositions[dict_indices[i_year]['pos']],
                    'Offset': offsets[dict_indices[i_year]['pos']],
                    'Year': years[dict_indices[i_year]['pos']], 'FileName': fileNames[dict_indices[i_year]['pos']]}

        data_neg = {'Content': contents[dict_indices[i_year]['neg']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['neg']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['neg']],
                    'Polarity': polarities[dict_indices[i_year]['neg']],
                    'LocalPosition': localPositions[dict_indices[i_year]['neg']],
                    'Offset': offsets[dict_indices[i_year]['neg']],
                    'Year': years[dict_indices[i_year]['neg']], 'FileName': fileNames[dict_indices[i_year]['neg']]
                    }

        data_neu = {'Content': contents[dict_indices[i_year]['neu']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['neu']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['neu']],
                    'Polarity': polarities[dict_indices[i_year]['neu']],
                    'LocalPosition': localPositions[dict_indices[i_year]['neu']],
                    'Offset': offsets[dict_indices[i_year]['neu']],
                    'Year': years[dict_indices[i_year]['neu']], 'FileName': fileNames[dict_indices[i_year]['neu']]
                    }

        total_samples = [len(dict_indices[i_year]['pos']), len(dict_indices[i_year]['neg']),
                         len(dict_indices[i_year]['neu'])]

        tmp_df_train, tmp_df_dev, tmp_df_test = divide_data_2(data_pos, data_neg, data_neu, total_samples)
        df_trains.append(tmp_df_train)
        df_devs.append(tmp_df_dev)
        df_tests.append(tmp_df_test)

    df_train = pd.concat(df_trains)
    # print(df_train.groupby('Polarity').count())
    df_train.to_csv(data_train_path, index=False)
    print(collections.Counter(df_train.Polarity.values))
    # df_train.hist("Offset")

    df_dev = pd.concat(df_devs)
    # print(df_dev.groupby('Polarity').count())
    df_dev.to_csv(data_dev_path, index=False)
    print(collections.Counter(df_dev.Polarity.values))

    df_test = pd.concat(df_tests)
    # print(df_test.groupby('Polarity').count())
    df_test.to_csv(data_test_path, index=False)
    print(collections.Counter(df_test.Polarity.values))


def divide_existing_test_data(data_file_path, data_train_path, data_dev_path, data_test_path):
    columns_names = ['Content', 'LineNumber', 'NamedEntity',
                     'Polarity', 'LocalPosition', 'Offset',
                     'Year', 'FileName']
    # data_file_path = input_folder + dataset + ".csv"
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
                     names=columns_names)

    df_test = pd.read_csv(data_test_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
                          names=columns_names)

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    contents = df.Content.values
    lineNumbers = df.LineNumber.values
    namedEntities = df.NamedEntity.values
    polarities = df.Polarity.values
    localPositions = df.LocalPosition.values
    offsets = df.Offset.values
    years = df.Year.values
    fileNames = df.FileName.values
    print('set(years) ', set(years))
    print(collections.Counter(df.Polarity.values))
    dict_indices = {}  # neu, pos, neg
    for i_year in set(years):
        dict_indices[i_year] = {}

    for i in range(len(polarities)):
        polar = polarities[i]
        year = years[i]
        if polar == 2:  # 'n':
            if 'neu' not in dict_indices[year]:
                dict_indices[year]['neu'] = [i]
            else:
                dict_indices[year]['neu'].append(i)
        elif polar == 0:  # '+':
            if 'pos' not in dict_indices[year]:
                dict_indices[year]['pos'] = [i]
            else:
                dict_indices[year]['pos'].append(i)
        elif polar == 1:  # '-':
            if 'neg' not in dict_indices[year]:
                dict_indices[year]['neg'] = [i]
            else:
                dict_indices[year]['neg'].append(i)

    df_trains, df_devs, df_tests = [], [], []
    df_pos, df_neg, df_neu = pd.DataFrame(columns=columns_names), pd.DataFrame(columns=columns_names), pd.DataFrame(
        columns=columns_names)
    for i_year in set(years):
        print(i_year, len(dict_indices[i_year]['pos']), len(dict_indices[i_year]['neg']),
              len(dict_indices[i_year]['neu']))

        data_pos = {'Content': contents[dict_indices[i_year]['pos']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['pos']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['pos']],
                    'Polarity': polarities[dict_indices[i_year]['pos']],
                    'LocalPosition': localPositions[dict_indices[i_year]['pos']],
                    'Offset': offsets[dict_indices[i_year]['pos']],
                    'Year': years[dict_indices[i_year]['pos']], 'FileName': fileNames[dict_indices[i_year]['pos']]}

        data_neg = {'Content': contents[dict_indices[i_year]['neg']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['neg']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['neg']],
                    'Polarity': polarities[dict_indices[i_year]['neg']],
                    'LocalPosition': localPositions[dict_indices[i_year]['neg']],
                    'Offset': offsets[dict_indices[i_year]['neg']],
                    'Year': years[dict_indices[i_year]['neg']], 'FileName': fileNames[dict_indices[i_year]['neg']]
                    }

        data_neu = {'Content': contents[dict_indices[i_year]['neu']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['neu']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['neu']],
                    'Polarity': polarities[dict_indices[i_year]['neu']],
                    'LocalPosition': localPositions[dict_indices[i_year]['neu']],
                    'Offset': offsets[dict_indices[i_year]['neu']],
                    'Year': years[dict_indices[i_year]['neu']], 'FileName': fileNames[dict_indices[i_year]['neu']]
                    }

        df_pos, df_neg, df_neu = merge_existing_test_data(data_pos, data_neg, data_neu, df_test, df_pos, df_neg, df_neu)
        # df_tests.append(tmp_df_test)

    part = 8
    list_pos_tmp, list_neg_tmp, list_neu_tmp = [], [], []
    tmp_pos_step, tmp_neg_step, tmp_neu_step = 10, 16, 1218

    count_part = 1
    for i in range(0, len(df_pos.FileName.values), tmp_pos_step):
        df_pos_dev = df_pos.iloc[i: i + tmp_pos_step]
        df_pos_train = df_pos.iloc[: i].append(df_pos.iloc[i + tmp_pos_step:])
        list_pos_tmp.append((df_pos_dev, df_pos_train))
        if count_part == part:
            break
        count_part += 1

    count_part = 1
    for i in range(0, len(df_neg.FileName.values), tmp_neg_step):
        df_neg_dev = df_neg.iloc[i: i + tmp_neg_step]
        df_neg_train = df_neg.iloc[: i].append(df_neg.iloc[i + tmp_neg_step:])
        list_neg_tmp.append((df_neg_dev, df_neg_train))
        if count_part == part:
            break
        count_part += 1

    # df_neu = df_neu.sample(frac=1)
    j = 0
    for i in range(len(list_pos_tmp)):
        df_neu = df_neu.sample(frac=1)
        df_pos_dev, df_pos_train = list_pos_tmp[i]
        df_neg_dev, df_neg_train = list_neg_tmp[i]
        df_neu_dev = df_neu.iloc[0: tmp_neu_step]
        df_neu_train = df_neu.iloc[
                       tmp_neu_step:tmp_neu_step + 20 * max([len(df_pos_train.index), len(df_neg_train.index)])]
        df_dev = pd.concat([df_pos_dev, df_neg_dev, df_neu_dev]).sample(
            frac=1)  # df_pos_dev.append(df_neg_dev).append(df_neu_dev).sample(frac=1)
        df_train = pd.concat([df_pos_train, df_neg_train, df_neu_train, df_pos_train, df_neg_train]).sample(
            frac=1)  # df_neu_train.append(df_pos_train).append(df_pos_train).append(df_neg_train).append(df_neg_train).sample(frac=1)
        j = j + tmp_neu_step
        print(collections.Counter(df_dev.Polarity.values))
        print(collections.Counter(df_train.Polarity.values))
        df_dev.to_csv(data_dev_path.replace('.csv', '_' + str(i) + '.csv'), index=False)
        df_train.to_csv(data_train_path.replace('.csv', '_' + str(i) + '.csv'), index=False)

        # break


def divide_new_data(data_file_path, data_train_path, data_dev_path, data_test_path):
    columns_names = ['Content', 'LineNumber', 'NamedEntity',
                     'Polarity', 'LocalPosition', 'Offset',
                     'Year', 'FileName']
    # data_file_path = input_folder + dataset + ".csv"
    df = pd.read_csv(data_file_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
                     names=columns_names)

    # df_test = pd.read_csv(data_test_path, delimiter=',', lineterminator='\n', encoding="utf-8", header=0,
    #                       names=columns_names)

    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    contents = df.Content.values
    lineNumbers = df.LineNumber.values
    namedEntities = df.NamedEntity.values
    polarities = df.Polarity.values
    localPositions = df.LocalPosition.values
    offsets = df.Offset.values
    years = df.Year.values
    fileNames = df.FileName.values
    print('set(years) ', set(years))
    print(collections.Counter(df.Polarity.values))
    dict_indices = {}  # neu, pos, neg
    for i_year in set(years):
        dict_indices[i_year] = {}

    for i in range(len(polarities)):
        polar = polarities[i]
        year = years[i]
        if polar == 2:  # 'n':
            if 'neu' not in dict_indices[year]:
                dict_indices[year]['neu'] = [i]
            else:
                dict_indices[year]['neu'].append(i)
        elif polar == 0:  # '+':
            if 'pos' not in dict_indices[year]:
                dict_indices[year]['pos'] = [i]
            else:
                dict_indices[year]['pos'].append(i)
        elif polar == 1:  # '-':
            if 'neg' not in dict_indices[year]:
                dict_indices[year]['neg'] = [i]
            else:
                dict_indices[year]['neg'].append(i)

    df_trains, df_devs, df_tests = [], [], []
    df_pos, df_neg, df_neu = pd.DataFrame(columns=columns_names), pd.DataFrame(columns=columns_names), pd.DataFrame(
        columns=columns_names)
    for i_year in set(years):
        print(i_year, len(dict_indices[i_year]['pos']), len(dict_indices[i_year]['neg']),
              len(dict_indices[i_year]['neu']))

        data_pos = {'Content': contents[dict_indices[i_year]['pos']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['pos']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['pos']],
                    'Polarity': polarities[dict_indices[i_year]['pos']],
                    'LocalPosition': localPositions[dict_indices[i_year]['pos']],
                    'Offset': offsets[dict_indices[i_year]['pos']],
                    'Year': years[dict_indices[i_year]['pos']], 'FileName': fileNames[dict_indices[i_year]['pos']]}

        data_neg = {'Content': contents[dict_indices[i_year]['neg']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['neg']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['neg']],
                    'Polarity': polarities[dict_indices[i_year]['neg']],
                    'LocalPosition': localPositions[dict_indices[i_year]['neg']],
                    'Offset': offsets[dict_indices[i_year]['neg']],
                    'Year': years[dict_indices[i_year]['neg']], 'FileName': fileNames[dict_indices[i_year]['neg']]
                    }

        data_neu = {'Content': contents[dict_indices[i_year]['neu']],
                    'LineNumber': lineNumbers[dict_indices[i_year]['neu']],
                    'NamedEntity': namedEntities[dict_indices[i_year]['neu']],
                    'Polarity': polarities[dict_indices[i_year]['neu']],
                    'LocalPosition': localPositions[dict_indices[i_year]['neu']],
                    'Offset': offsets[dict_indices[i_year]['neu']],
                    'Year': years[dict_indices[i_year]['neu']], 'FileName': fileNames[dict_indices[i_year]['neu']]
                    }

        df_pos = pd.concat([df_pos, pd.DataFrame(data_pos)])  # .sample(frac = 1)
        df_neg = pd.concat([df_neg, pd.DataFrame(data_neg)])  # pd.DataFrame(data_neg)  # .sample(frac = 1)
        df_neu = pd.concat([df_neu, pd.DataFrame(data_neu)])  # pd.DataFrame(data_neu)  # .sample(frac = 1)

    part = 5
    list_pos_tmp, list_neg_tmp, list_neu_tmp = [], [], []
    tmp_pos_step, tmp_neg_step, tmp_neu_step = int(0.1 * len(df_pos)), int(0.1 * len(df_neg)), int(0.1 * len(df_neu))

    count_part = 1
    for i in range(0, len(df_pos.FileName.values), tmp_pos_step):
        df_pos_dev = df_pos.iloc[i: i + tmp_pos_step]
        df_pos_test = df_pos.iloc[i + tmp_pos_step: i + 2 * tmp_pos_step]
        df_pos_train = df_pos.iloc[: i].append(df_pos.iloc[i + 2 * tmp_pos_step:])
        list_pos_tmp.append((df_pos_dev, df_pos_test, df_pos_train))
        if count_part == part:
            break
        count_part += 1

    count_part = 1
    for i in range(0, len(df_neg.FileName.values), tmp_neg_step):
        df_neg_dev = df_neg.iloc[i: i + tmp_neg_step]
        df_neg_test = df_neg.iloc[i + tmp_neg_step: i + 2 * tmp_neg_step]
        df_neg_train = df_neg.iloc[: i].append(df_neg.iloc[i + 2 * tmp_neg_step:])
        list_neg_tmp.append((df_neg_dev, df_neg_test, df_neg_train))
        if count_part == part:
            break
        count_part += 1

    # df_neu = df_neu.sample(frac=1)
    j = 0
    for i in range(len(list_pos_tmp)):
        df_neu = df_neu.sample(frac=1)
        df_pos_dev, df_pos_test, df_pos_train = list_pos_tmp[i]
        df_neg_dev, df_neg_test, df_neg_train = list_neg_tmp[i]
        df_neu_dev = df_neu.iloc[0: tmp_neu_step]
        df_neu_test = df_neu.iloc[tmp_neu_step: 2 * tmp_neu_step]
        df_neu_train = df_neu.iloc[
                       2 * tmp_neu_step:2 * tmp_neu_step + 20 * max([len(df_pos_train.index), len(df_neg_train.index)])]
        df_dev = pd.concat([df_pos_dev, df_neg_dev, df_neu_dev]).sample(
            frac=1)  # df_pos_dev.append(df_neg_dev).append(df_neu_dev).sample(frac=1)
        df_test = pd.concat([df_pos_test, df_neg_test, df_neu_test]).sample(
            frac=1)
        df_train = pd.concat([df_pos_train, df_neg_train, df_neu_train, df_pos_train, df_neg_train]).sample(
            frac=1)  # df_neu_train.append(df_pos_train).append(df_pos_train).append(df_neg_train).append(df_neg_train).sample(frac=1)
        j = j + tmp_neu_step

        print(collections.Counter(df_dev.Polarity.values))
        print(collections.Counter(df_test.Polarity.values))
        print(collections.Counter(df_train.Polarity.values))
        df_dev.to_csv(data_dev_path.replace('.csv', '_' + str(i) + '.csv'), index=False)
        df_test.to_csv(data_test_path.replace('.csv', '_' + str(i) + '.csv'), index=False)
        df_train.to_csv(data_train_path.replace('.csv', '_' + str(i) + '.csv'), index=False)

        # break


def divide_data_2(data_pos, data_neg, data_neu, total_samples):
    # Create DataFrame
    df_pos_shuffle = pd.DataFrame(data_pos)  # .sample(frac = 1)
    df_neg_shuffle = pd.DataFrame(data_neg)  # .sample(frac = 1)
    df_neu_shuffle = pd.DataFrame(data_neu)  # .sample(frac = 1)

    num_dev_samples = [round(0.1 * i) for i in total_samples]

    df_dev = df_pos_shuffle.iloc[:num_dev_samples[0]].append(df_neg_shuffle.iloc[:num_dev_samples[1]]).append(
        df_neu_shuffle.iloc[:num_dev_samples[2]]).sample(frac=1)

    df_test = df_pos_shuffle.iloc[num_dev_samples[0]:2 * num_dev_samples[0]].append(
        df_neg_shuffle.iloc[num_dev_samples[1]:2 * num_dev_samples[1]]).append(
        df_neu_shuffle.iloc[num_dev_samples[2]:2 * num_dev_samples[2]]).sample(frac=1)

    # lessdata
    len_train_pos = len(df_pos_shuffle.iloc[:2 * num_dev_samples[0]])
    len_train_neg = len(df_neg_shuffle.iloc[:2 * num_dev_samples[1]])
    len_train_neu = len(df_neu_shuffle.iloc[:2 * num_dev_samples[2]])

    df_train_pos = df_pos_shuffle.iloc[len_train_pos:]  # .append(df_pos_shuffle.iloc[len_train_pos:])
    df_train_neg = df_neg_shuffle.iloc[len_train_neg:]  # .append(df_neg_shuffle.iloc[len_train_neg:])
    num_pos_neg = max([len(df_train_pos), len(df_train_neg)])
    df_train_neu = df_neu_shuffle.iloc[len_train_neu:].sample(frac=1)[:int(num_pos_neg * 10)]
    df_train = df_train_pos.append(df_train_neg).append(df_train_neu).sample(frac=1)

    print('*****Statistical:  pos, neg, neu, total****')
    print('Dev: ', len(df_pos_shuffle.iloc[num_dev_samples[0]:2 * num_dev_samples[0]]),
          len(df_neg_shuffle.iloc[:num_dev_samples[1]]), len(df_neu_shuffle.iloc[:num_dev_samples[2]]), len(df_dev))
    print('Test: ', len(df_pos_shuffle.iloc[num_dev_samples[0]:2 * num_dev_samples[0]]),
          len(df_neg_shuffle.iloc[num_dev_samples[1]:2 * num_dev_samples[1]]),
          len(df_neu_shuffle.iloc[num_dev_samples[2]:2 * num_dev_samples[2]]), len(df_test))
    print('Train: ', len(df_train_pos), len(df_train_neg), len(df_train_neu), len(df_train))

    return df_train, df_dev, df_test


def filter_lines(df_pos_shuffle, df_test, df_pos):
    df_pos_shuffle['Key'] = df_pos_shuffle['FileName'].str.cat(df_pos_shuffle['LineNumber'].astype(str), sep=";")
    test_keys = set(df_test.Key.values)
    pos_keys = df_pos_shuffle.Key.values

    for i in range(len(pos_keys)):
        if pos_keys[i] not in test_keys:
            df_pos = df_pos.append(df_pos_shuffle.iloc[i], ignore_index=True)

    # df_pos = pd.concat([pd.DataFrame(df_pos_shuffle.iloc[i]) for i in range(len(pos_keys)) if pos_keys[i] not in test_keys], ignore_index=True)

    return df_pos


def filter_new_lines(df_pos_shuffle, df_pos):
    df_pos_shuffle['Key'] = df_pos_shuffle['FileName'].str.cat(df_pos_shuffle['LineNumber'].astype(str), sep=";")
    # test_keys = set(df_test.Key.values)
    pos_keys = df_pos_shuffle.Key.values

    for i in range(len(pos_keys)):
        # if pos_keys[i] not in test_keys:
        df_pos = df_pos.append(df_pos_shuffle.iloc[i], ignore_index=True)

    # df_pos = pd.concat([pd.DataFrame(df_pos_shuffle.iloc[i]) for i in range(len(pos_keys)) if pos_keys[i] not in test_keys], ignore_index=True)

    return df_pos


def merge_existing_test_data(data_pos, data_neg, data_neu, df_test, df_pos, df_neg, df_neu):
    # Create DataFrame
    df_pos_shuffle = pd.DataFrame(data_pos)  # .sample(frac = 1)
    df_neg_shuffle = pd.DataFrame(data_neg)  # .sample(frac = 1)
    df_neu_shuffle = pd.DataFrame(data_neu)  # .sample(frac = 1)


    df_test['Key'] = df_test['FileName'].str.cat(df_test['LineNumber'].astype(str), sep=";")

    df_pos = filter_lines(df_pos_shuffle, df_test, df_pos)
    df_neg = filter_lines(df_neg_shuffle, df_test, df_neg)
    df_neu = filter_lines(df_neu_shuffle, df_test, df_neu)

    return df_pos, df_neg, df_neu


def merge_new_data(data_pos, data_neg, data_neu, df_pos, df_neg, df_neu):
    # Create DataFrame
    df_pos_shuffle = pd.DataFrame(data_pos)  # .sample(frac = 1)
    df_neg_shuffle = pd.DataFrame(data_neg)  # .sample(frac = 1)
    df_neu_shuffle = pd.DataFrame(data_neu)  # .sample(frac = 1)

 
    df_pos = filter_new_lines(df_pos_shuffle, df_pos)
    df_neg = filter_new_lines(df_neg_shuffle, df_neg)
    df_neu = filter_new_lines(df_neu_shuffle, df_neu)

    return df_pos, df_neg, df_neu


def test_split(file_path):
    text = open(file_path, 'r').read().split("\n")
    print(len(text))


def main():
    flag_server = True
    data_folder = "../"

    if flag_server:
        parser = argparse.ArgumentParser()
        parser.add_argument("curr_lang", help="Choose language", type=str)  # Finnish, French, German, Swedish
        parser.add_argument("type_process",
                            help="pre-process (pre) or post-process (post) or combine (comb) or pre-train",
                            type=str)  # Finnish, French, German, Swedish
        # parser.add_argument("cased", help="Choose cased", type=int)  # 1: cased
        args = parser.parse_args()

        curr_lang = args.curr_lang  # German
        type_process = args.type_process
        # flag_cased = args.cased
        data_test_path = data_folder + curr_lang + "_data_test_input/"
        data_train_path = data_folder + curr_lang + "_data_train_input/"
        data_dev_path = data_folder + curr_lang + "_data_dev_input/"
    else:
        curr_lang = 'Finnish'  # 'German'  # German 'Finnish'
        type_process = 'pre2021'  # 'divide_new_data' # 'divide_data'  # 'pre_train'#
        data_test_path = data_folder + curr_lang + "_data_test_input/"
        data_train_path = data_folder + curr_lang + "_data_train_input/"
        data_dev_path = data_folder + curr_lang + "_data_dev_input/"

    flag_cased = 1
    max_seq_len = 128  # args.max_seq_len

    if type_process.lower() == 'pre':
        txt2csv(data_test_path)
    elif type_process.lower() == 'pre2021':
        txt2csv_2021(data_train_path)
        txt2csv_2021(data_dev_path)
        txt2csv_2021(data_test_path)
    elif type_process.lower() == 'post':
        write_txt_output(data_test_path, flag_cased, curr_lang)

    elif type_process.lower() == 'comb':
        nel_folder = data_test_path + curr_lang + "_nel/"
        stance_folder = data_test_path + "output_cased_3/"
        nel_sd_folder = data_test_path + curr_lang + "_nel_sd/"
        print(nel_folder)
        print(stance_folder)
        print(nel_sd_folder)
        if not os.path.exists(nel_sd_folder):
            os.mkdir(nel_sd_folder)
        combine_nel_empty_stance(nel_folder, stance_folder, nel_sd_folder)

    elif type_process.lower() == 'pre_train':
        csv_file_path = data_test_path + curr_lang + "_data_all.csv"
        txt2csv_train_dev_beg(data_test_path, csv_file_path)

    elif type_process.lower() == 'divide_data':
        csv_file_path = data_test_path + curr_lang + "_data_all.csv"
        train_folder = data_folder + curr_lang + "_data_train_input/"
        dev_folder = data_folder + curr_lang + "_data_dev_input/"
        test_folder = data_folder + curr_lang + "_data_test_input/"
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        if not os.path.exists(dev_folder):
            os.mkdir(dev_folder)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)

        train_file_path = train_folder + curr_lang + "_data_train_v5.csv"
        dev_file_path = dev_folder + curr_lang + "_data_dev_v5.csv"
        test_file_path = test_folder + curr_lang + "_data_test_v5.csv"
        # divide_existing_test_data(data_file_path, data_train_path, data_dev_path, data_test_path)
        divide_existing_test_data(csv_file_path, train_file_path, dev_file_path, test_file_path)
        # divide_data(csv_file_path, train_file_path, dev_file_path, test_file_path)

    elif type_process.lower() == 'divide_new_data':
        csv_file_path = data_test_path + curr_lang + "_data_all.csv"
        train_folder = data_folder + curr_lang + "_data_train_input/"
        dev_folder = data_folder + curr_lang + "_data_dev_input/"
        test_folder = data_folder + curr_lang + "_data_test_input/"
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        if not os.path.exists(dev_folder):
            os.mkdir(dev_folder)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)

        train_file_path = train_folder + curr_lang + "_data_train_v5.csv"
        dev_file_path = dev_folder + curr_lang + "_data_dev_v5.csv"
        test_file_path = test_folder + curr_lang + "_data_test_v5.csv"
        # divide_existing_test_data(data_file_path, data_train_path, data_dev_path, data_test_path)
        # divide_existing_test_data(csv_file_path, train_file_path, dev_file_path, test_file_path)
        divide_new_data(csv_file_path, train_file_path, dev_file_path, test_file_path)


if __name__ == '__main__':
    main()
