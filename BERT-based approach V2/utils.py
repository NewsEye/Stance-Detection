'''
Created on 13 Jul 2020

@author: HaiNguyen
'''
import codecs, json
import pandas as pd
import argparse
from openpyxl import load_workbook


def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))


def read_dict(dict_file_path):
    with codecs.open(dict_file_path, encoding='utf-8') as myfile:
        dict_tmp = json.load(myfile)
    return dict_tmp


def writeExcel(excelPath, excelHeader, listData, sheet_name='Sheet1', engine='xlsxwriter'):
    df = pd.DataFrame(listData, columns=excelHeader)

    writer = pd.ExcelWriter(excelPath, engine)
    if engine == 'openpyxl':
        writer.book = load_workbook(excelPath)
    df.to_excel(writer, sheet_name, encoding="utf-8")
    writer.save()


def collect_results_lexicon(list_params):
    output_file_path = 'all_results_lexicon.xlsx'
    languages = ["Finnish", 'French', 'German', 'Swedish']
    list_params = list_params.split(',')
    paras = [(list_params[i], list_params[i + 1]) for i in range(0, len(list_params), 2)]
    paras = [(1, 0.1), (1, 0.1), (2, 0.1), (1, 0.1)]
    fieldnames = ['Language', 'Cased', 'Stance', 'Precision', 'Recall', 'F-score', 'Support']
    list_data = []
    for i in range(len(languages)):
        lang = languages[i]
        para = paras[i]
        path = "../" + lang + '_data_test_input/results/'
        file_name = "test_result_dict_" + str(para[0]) + "_" + str(para[1]) + ".json"

        dict_result = read_dict(path + file_name)
        list_data.append([lang, i, 'POS',
                          dict_result['0']['precision'], dict_result['0']['recall'], dict_result['0']['f1-score'],
                          dict_result['0']['support']])
        list_data.append([lang, i, 'NEG',
                          dict_result['1']['precision'], dict_result['1']['recall'], dict_result['1']['f1-score'],
                          dict_result['1']['support']])
        list_data.append([lang, i, 'NEU',
                          dict_result['2']['precision'], dict_result['2']['recall'], dict_result['2']['f1-score'],
                          dict_result['2']['support']])
        list_data.append([lang, i, 'Accuracy',
                          dict_result['accuracy'], 0, 0, 0])
        list_data.append([lang, i, 'F1-macro',
                          dict_result['macro avg']['f1-score'], 0, 0, dict_result['macro avg']['support']])

    writeExcel(output_file_path, fieldnames, list_data)
    print('Finnish collecting, please check file ', output_file_path)


def collect_result_BERT(list_epochs):
    languages = ["Finnish", 'French', 'German', 'Swedish']
    #     epochs = [(4,2), (3, 9), (2,4), (9,9)]
    list_epochs = list_epochs.split(',')
    #     print(list_epochs)
    epochs = [(list_epochs[i + 1], list_epochs[i]) for i in range(0, len(list_epochs), 2)]
    #     print(epochs)
    # epochs = [(9, 3), (7, 6), (1, 2), (9, 9)]  # lessdata
    #     epochs = [(9,7), (7,9), (1,1), (4,5)]#fulldata

    output_file_path = 'all_results_halfdata.xlsx'
    fieldnames = ['Language', 'Cased', 'Stance', 'Precision', 'Recall', 'F-score', 'Support']
    #     stance_vals = {'0':'POS','1':'NEG','2':'NEU'}
    list_data = []
    for i in range(len(languages)):
        lang = languages[i]
        path = "../" + lang + '_data_test_input/results/'
        epoch = epochs[i]
        file_names = ['test_result_uncased_' + str(epoch[0]) + '.json', 'test_result_cased_' + str(epoch[1]) + '.json']
        for j in range(len(file_names)):
            file_name = file_names[j]
            dict_result = read_dict(path + file_name)
            list_data.append([lang, j, 'POS',
                              dict_result['0']['precision'], dict_result['0']['recall'], dict_result['0']['f1-score'],
                              dict_result['0']['support']])
            list_data.append([lang, j, 'NEG',
                              dict_result['1']['precision'], dict_result['1']['recall'], dict_result['1']['f1-score'],
                              dict_result['1']['support']])
            list_data.append([lang, j, 'NEU',
                              dict_result['2']['precision'], dict_result['2']['recall'], dict_result['2']['f1-score'],
                              dict_result['2']['support']])
            list_data.append([lang, j, 'Accuracy',
                              dict_result['accuracy'], 0, 0, 0])
            list_data.append([lang, j, 'F1-macro',
                              dict_result['macro avg']['f1-score'], 0, 0, dict_result['macro avg']['support']])

    writeExcel(output_file_path, fieldnames, list_data)
    print('Finnish collecting, please check file ', output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("approach_type", help="Which approach type? Lexical or BERT", type=str)
    parser.add_argument("list_params",
                        help="BERT epochs or (#context, threshold) of each language."
                             "E.g. BERT epochs: 3,4,5,6,7,8,9,10 corresponds "
                             "Finnish (cased, uncased), French (cased, uncased), "
                             "German (cased, uncased), Swedish (cased, uncased)",
                        type=str)

    args = parser.parse_args()
    approach_type = args.approach_type
    list_params = args.list_params
    if approach_type.lower() == 'lexical':
        collect_results_lexicon(list_params)
    else:
        collect_result_BERT(list_params)
