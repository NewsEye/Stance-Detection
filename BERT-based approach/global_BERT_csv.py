#************* Library
#!/usr/bin/env python
# coding: utf-8
# FINAL RESULTS 20200226_1830

from googletrans import Translator
from nltk.corpus import sentiwordnet as swn
import codecs, re
import os, string, json
import spacy
from stop_words import get_stop_words
import xml.dom.minidom    
import sklearn.metrics as metrics
import unicodedata
import csv, pandas

#************* Global variables
#--server
data_folder = "/hainguyen/STANCE_DETECTION/"
dataset = "PULS_data_dev"
pre_input_folder = data_folder+dataset+"/"
input_folder = data_folder+dataset+"_input/"
output_folder = data_folder+dataset+"_output/" 

if not os.path.exists(input_folder):
    os.mkdir(input_folder)
    
if not os.path.exists(output_folder):
    os.mkdir(output_folder)    


results_folder=data_folder+dataset+"_results/"    
    
lang = 'en'#'de';#'fr'
translator = Translator()

num_context_word = 5
senti_folder = "../senti_lexicon/"
senti_trans_folder = "../senti_lexicon_trans/"
senti_file_path = "senti_dict.json"

# nlp = spacy.load(lang+'_core_news_md')
stop_words = set(get_stop_words(lang))
xml_bilexicon_path = "../XML_translation/"
puncts = "—"

#************* List of functions
def convert_file_dicts(file_name):
    dict_toks = {}
    dict_NEs = {}
    dict_global_local_index = {}
    dict_lines_words = {}
    
    if file_name not in dict_toks:
        dict_toks[file_name] = []
        dict_NEs[file_name] = []
        dict_lines_words[file_name] = []
        dict_global_local_index[file_name] = {}
    file_path = input_folder + file_name
    text = codecs.open(file_path, 'rb', encoding="utf-8").read().split("\n")
    tok_index = 0
    tmp_NE = ""; tmp_NE_index = -1; tmp_NE_offset = 1
    count_line = 0
    for line in text:
        list_line_tokens = []
        if line.strip() == '': 
#             print("######"); 
            count_line += 1; 
            dict_lines_words[file_name].append(list_line_tokens)
#             if tmp_NE != "":
#                 tmp_NE_upper = tmp_NE.strip().upper()
#                 dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset), tmp_NE_upper))            
            continue
            
        lastTokenPos = 0
        local_tok_index = 0
        
        for spacePos in re.finditer(r"$|\ ", line):
            tokenEndPos = spacePos.start()
            tokenNE = line[lastTokenPos:tokenEndPos]
            list_line_tokens.append(tokenNE)
            
            print(tokenNE)
            token = tokenNE.split("__")[0]
            tag_NE = tokenNE.split("__")[1]
            if len(tag_NE)>0:
    
                if tag_NE[0] == 'B':
                    if tmp_NE != "":
                        tmp_NE_upper = tmp_NE.strip()#.upper()
                        dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset), tmp_NE_upper))

    #                 else:        
                    tmp_NE = token + " "; tmp_NE_offset = 1; tmp_NE_index = tok_index


                elif tag_NE[0] == 'O':
                    if tmp_NE != "":
                        tmp_NE_upper = tmp_NE.strip()#.upper(); 
                        dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset), tmp_NE_upper))

                    tmp_NE = ""; tmp_NE_offset = 1; tmp_NE_index = -1

                elif tag_NE[0] == 'I' and tmp_NE != '':
                    tmp_NE += token + " "; tmp_NE_offset += 1
    
                
            lastTokenPos = tokenEndPos + 1       
            dict_toks[file_name].append((tok_index, token)) 
            dict_global_local_index[file_name][tok_index] = (count_line, local_tok_index)
                     
            local_tok_index += 1; tok_index += 1

#         if tmp_NE != "":
#             tmp_NE_upper = tmp_NE.strip().upper()
#             dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset), tmp_NE_upper))
                                
        count_line += 1
        dict_lines_words[file_name].append(list_line_tokens)
    
    if tmp_NE != "":
        tmp_NE_upper = tmp_NE.strip()#.upper()
        dict_NEs[file_name].append((str(tmp_NE_index) + ":" + str(tmp_NE_offset), tmp_NE_upper))   
          
    return dict_toks, dict_NEs, dict_global_local_index, dict_lines_words

def translate_sentence(translator, src_txt, src_lang='fr', tgt_lang='en'):
    translation = translator.translate(src_txt, dest=tgt_lang, src=src_lang)
    print(translation.origin, ' -> ', translation.text)
    return translation.text

def get_sentiment(word, senti_dict, src_lang='fr'):
    pos_score, neg_score = 0, 0
    key_len = str(len(word))
    if src_lang in senti_dict and key_len in senti_dict[src_lang] and word in senti_dict[src_lang][key_len]:
            pos_score, neg_score = senti_dict[src_lang][key_len][word]
    
    print(word, pos_score, neg_score)
    return pos_score, neg_score
    
    
    
#     word = 'crazy'
#     list_senti = list(swn.senti_synsets(word))
#     sum_pos = 0; sum_neg = 0
#     
#     for senti in list_senti:
#         sum_pos += senti.pos_score(); sum_neg += senti.neg_score()
#         
# 
#     avg_pos = sum_pos * 1.0 / len(list_senti) if len(list_senti) > 0 else 0
#     avg_neg = sum_neg * 1.0 / len(list_senti) if len(list_senti) > 0 else 0
#     
#     print(''.join([word,' (', str(round(avg_pos,3)),', ', str(round(avg_neg,3)),')']))  
#           
#     return avg_pos, avg_neg

def get_senti_sent(list_sent, lang):
    '''
    1: POS
    -1: NEG
    0: NEU

    '''
    src_txt = ' '.join([tok[1].lower().strip(string.punctuation).strip(puncts) for sent in list_sent for tok in sent])
    print('len_src_txt ', len([tok[1] for sent in list_sent for tok in sent]))
    
#     sent_trans = translate_sentence(translator, src_txt)
    print("sent ", src_txt)
#     print('sent_trans ', sent_trans)
    list_sentiments = []
#     doc = nlp(u"voudrais non animaux yeux dors couvre.")
#     for token in doc:
#         print(token, token.lemma_)    
#     list_tok_lemma = [tok.lemma_ for tok in nlp(src_txt)]
    if lang=='en':
        
        for tok in src_txt.split():
            
            list_sentiments.append(get_sentiment_wordnet(tok))
    else:
        
        for tok in src_txt.split():
            
            list_sentiments.append(get_sentiment(tok, senti_dict, lang))
                    
    assert len(list_sentiments) > 0, '%d' % (len(list_sentiments))
    number_bearing_words = len([i for i, j in list_sentiments if (i >0 or j>0)])
    if number_bearing_words >0: 
        avg_pos = round(1.0 * sum([i for i, _ in list_sentiments]) / number_bearing_words, 1) 
        avg_neg = round(1.0 * sum([j for _, j in list_sentiments]) / number_bearing_words, 1)
    else:
        avg_pos = avg_neg = 0.0
    print(avg_pos, avg_neg)
    
    if avg_pos > avg_neg: return '+'
    if avg_pos < avg_neg: return '-'
    if avg_pos == avg_neg: return 'n'
            
#     if avg_pos > 0.25 and avg_neg > 0.25: 
#         if avg_pos > avg_neg: return 1
#         if avg_pos < avg_neg: return -1
#         else: return 0
#     elif avg_pos > 0.25: return 1
#     elif avg_neg > 0.25: return -1
#     else: return 0
#     return max(avg_pos, avg_neg, 1-(avg_pos+avg_neg))

def create_NEs_senti(file_name, dict_NEs, dict_toks, dict_NEs_senti, lang): 
    
    if file_name not in dict_NEs_senti:
        dict_NEs_senti[file_name] = []
    
    list_context_word = []
    
    for ne_item in dict_NEs[file_name]:
        ne_pos = int(ne_item[0].split(":")[0])
        ne_offset = int(ne_item[0].split(":")[1])
        ne_val = ne_item[1]
#         if ne_pos == 11960: 
#             print()
        count_tok = 0; 
        
        if dict_toks[file_name][: ne_pos] != None and len(dict_toks[file_name][: ne_pos]) > 0:
            for tok in dict_toks[file_name][: ne_pos][::-1]:
                tok_clean = tok[1].lower().strip(string.punctuation).strip(puncts)
                if len(tok_clean) > 0 and tok_clean not in stop_words:
                    list_context_word.append(tok); count_tok += 1
                    if count_tok >= num_context_word: 
                        break
        count_tok = 0
        if dict_toks[file_name][ne_pos + ne_offset:] != None:
            for tok in dict_toks[file_name][ne_pos + ne_offset:]:
                tok_clean = tok[1].lower().strip(string.punctuation).strip(puncts)
                if len(tok_clean) > 0 and tok_clean  not in stop_words:
                    list_context_word.append(tok); count_tok += 1
                    if count_tok >= num_context_word: 
                        break
        
#         list_context_word = dict_toks[file_name][ne_pos - num_context_word: ne_pos]\
#         + dict_toks[file_name][ne_pos + ne_offset: ne_pos + ne_offset + num_context_word]        
        
#         if list_context_word != None and len(list_context_word) > 0:            
#             print(ne_val, ne_pos, ne_offset)
#             ne_senti = get_senti_sent([list_context_word], lang)
#             dict_NEs_senti[file_name].append((ne_item[0], ne_senti))
    return list_context_word

def create_file_senti(dict_pred, file_name, lang):
    dict_toks, dict_NEs, dict_global_local_index, dict_lines_words = convert_file_dicts(file_name)
    

    dict_NEs_senti = {}
#     dict_NEs_senti = create_NEs_senti(file_name, dict_NEs, dict_toks, dict_NEs_senti, lang)

    list_context_word = create_NEs_senti(file_name, dict_NEs, dict_toks, dict_NEs_senti, lang)
    
    ne_senti=""
    if list_context_word != None and len(list_context_word) > 0: 
        ne_senti = get_senti_sent([list_context_word], lang) 
    print(file_name) 
    if ne_senti=="": dict_pred[file_name] = "n"
    else: dict_pred[file_name] = ne_senti
    return dict_pred
#     for ne_senti in dict_NEs_senti[file_name]:
#         
#         itok, ioffset = ne_senti[0].split(':')  
#         itok, ioffset = int(itok), int(ioffset)
#         
#         for count_offset in range(0, ioffset):
#             (iline, iword) = dict_global_local_index[file_name][itok + count_offset]
#             
#             dict_lines_words[file_name][iline][iword] += "__"+str(ne_senti[1])
#             
#             if ne_senti[1] == 0: 
#                 dict_lines_words[file_name][iline][iword] += '__NEU'
#             elif ne_senti[1] == 1: 
#                 dict_lines_words[file_name][iline][iword] += '__POS'
#             else: 
#                 dict_lines_words[file_name][iline][iword] += '__NEG'   
# 
#             
#     list_lines = []
#     for line in dict_lines_words[file_name]:
#         list_lines.append(' '.join(line))
#     
#        
#     output_file_path = output_folder + file_name
#     print(output_file_path)
#     codecs.open(output_file_path, "wb", encoding='utf-8').write('\n'.join(list_lines))
    
def write_dict(dict_tmp, dict_file_path):
    with codecs.open(dict_file_path, 'wb', encoding='utf-8') as myfile:
        myfile.write(json.dumps(dict_tmp, indent=4, sort_keys=True))

def read_dict(dict_file_path):
    
    with codecs.open(dict_file_path, encoding='ascii') as myfile:
        dict_tmp = json.load(myfile)
    return dict_tmp  

def create_senti_dict(senti_dict, lang='fr'):
    '''
    GLOBAL: senti_folder, senti_file_path
    '''
    senti_flags = ['nega', 'posi']
   
    for senti_flag in senti_flags:
        pos_score, neg_score = 0, 0
        if senti_flag == 'nega':
            neg_score = 1.0
        elif senti_flag == 'posi':
            pos_score = 1.0
            
        senti_neg_file_name = senti_folder + senti_flag + "tive_words_" + lang + '.txt'
        
        text = codecs.open(senti_neg_file_name, 'rb', encoding='utf-8').read().strip().split("\n")
        if lang not in senti_dict:
            senti_dict[lang] = {}
#         if senti_flag not in senti_dict:
#             senti_dict[lang][senti_flag] = {}  
        for tok in text:
            tok = tok.strip()
            key_len = str(len(tok))
            if key_len not in senti_dict[lang]:
                senti_dict[lang][key_len] = {tok:(pos_score, neg_score)}
            else:
                if tok not in senti_dict[lang][key_len]:
                    senti_dict[lang][key_len][tok] = (pos_score, neg_score)
                else:                    
                    (pos_score, neg_score) = senti_dict[lang][key_len][tok]
                    if senti_flag == 'nega':
                        neg_score += 1.0
                    elif senti_flag == 'nega':
                        pos_score += 1.0                
                    senti_dict[lang][key_len][tok] = (pos_score, neg_score)
    
    write_dict(senti_dict, senti_file_path) 



def create_senti_dict_translation(senti_dict):
    '''
    GLOBAL: xml_bilexicon_path, senti_file_path
    '''
    
    for file_name in os.listdir(xml_bilexicon_path):
        if '.ipynb_checkpoints' in file_name: continue
        file_path = xml_bilexicon_path + file_name        
        lang_tmp = file_name[:2]
        if lang_tmp not in senti_dict:
            senti_dict[lang_tmp] = {}        
        # use the parse() function to load and parse an XML file
        doc = xml.dom.minidom.parse(file_path)
    
        # print out the document node and the name of the first child tag
        print ('doc.nodeName ', doc.nodeName)
        
        # get a list of XML tags from the document and print each one
        entries = doc.getElementsByTagName("entry")
        print ("%d entries:" % entries.length)
        for entry in entries:
            orths = entry.getElementsByTagName('orth')
            for orth in orths:  # only one orth
                tok = orth.firstChild.nodeValue
                print(tok)
            
            pos_score, neg_score = 0, 0    
            quotes = entry.getElementsByTagName('quote')
            for quote in quotes:
                en_words = quote.firstChild.nodeValue.split()                
                print('**', en_words)
                for en_word in en_words:            
                    pos_score_tmp, neg_score_tmp = get_sentiment_wordnet(en_word)
                    pos_score += pos_score_tmp
                    neg_score += neg_score_tmp
            pos_score = pos_score / len(quotes)
            neg_score = neg_score / len(quotes)
            
            if pos_score > 0 or neg_score > 0:    
                key_len = str(len(tok))
                if key_len not in senti_dict[lang_tmp]:
                    senti_dict[lang_tmp][key_len] = {tok:(pos_score, neg_score)}
                else:
                    if tok not in senti_dict[lang_tmp][key_len]:
                        senti_dict[lang_tmp][key_len][tok] = (pos_score, neg_score)
                    else:                        
                        (pos_score_tmp, neg_score_tmp) = senti_dict[lang_tmp][key_len][tok]
                        senti_dict[lang_tmp][key_len][tok] = (pos_score + pos_score_tmp, neg_score + neg_score_tmp)
                            
    write_dict(senti_dict, senti_file_path) 
    

def create_senti_dict_googletrans(senti_dict):
    '''
    GLOBAL: xml_bilexicon_path, senti_file_path, senti_trans_folder
    file is translated by google translate before compute NE sentiment
    
    '''
    
    for file_name in os.listdir(senti_folder):
        if '.ipynb_checkpoints' in file_name: continue
        file_path = senti_folder + file_name        
        lang_tmp = file_name.split('.txt')[0][-2:]
        print(lang_tmp)
        if lang_tmp not in senti_dict:
            senti_dict[lang_tmp] = {}        
            
        text = codecs.open(file_path, 'rb', encoding='utf-8').read().split('\n')
        text_trans = codecs.open(senti_trans_folder + file_name, 'rb', encoding='utf-8').read().split('\n')
        
        toks = [tok.strip() for tok in text]
        toks_trans = [tok_trans.strip() for tok_trans in text_trans]
        
        for i in range(len(toks_trans)):     
            tok = toks[i]
                  
            en_words = toks_trans[i].split()
            for en_word in en_words:
                pos_score, neg_score = 0, 0
                pos_score_tmp, neg_score_tmp = get_sentiment_wordnet(en_word)
                pos_score += pos_score_tmp; neg_score += neg_score_tmp
             
            if pos_score > 0 or neg_score > 0:    
                key_len = str(len(tok))
                if key_len not in senti_dict[lang_tmp]:
                    senti_dict[lang_tmp][key_len] = {tok:(pos_score, neg_score)}
                else:
                    if tok not in senti_dict[lang_tmp][key_len]:
                        senti_dict[lang_tmp][key_len][tok] = (pos_score, neg_score)
                    else:                        
                        (pos_score_tmp, neg_score_tmp) = senti_dict[lang_tmp][key_len][tok]
                        senti_dict[lang_tmp][key_len][tok] = (pos_score + pos_score_tmp, neg_score + neg_score_tmp)
                         
    write_dict(senti_dict, senti_file_path) 
    
def get_sentiment_wordnet(word):
#     word = 'crazy'
    list_senti = list(swn.senti_synsets(word))
    sum_pos = 0; sum_neg = 0
    
    for senti in list_senti:
        sum_pos += senti.pos_score(); sum_neg += senti.neg_score()
        

    avg_pos = sum_pos * 1.0 / len(list_senti) if len(list_senti) > 0 else 0
    avg_neg = sum_neg * 1.0 / len(list_senti) if len(list_senti) > 0 else 0
    
    print(''.join([word, ' (', str(round(avg_pos, 3)), ', ', str(round(avg_neg, 3)), ')']))  
          
    return avg_pos, avg_neg     
              

# import sklearn.metrics 
def create_input():
    '''
    convert from pre_input into input (correct format suggested by Ahmed)
    global variables: pre_input_folder, input_folder
    '''
    for file_name in os.listdir(pre_input_folder):
        print(file_name)
        if '.ipynb_checkpoints' in file_name: continue
        file_path = pre_input_folder + file_name
        input_file_path = input_folder + file_name
        texts = codecs.open(file_path, 'r', encoding='utf-8').read().split('\n')
         
        output_texts = codecs.open(input_file_path, 'wb', encoding='utf-8')

        for line in texts:
            if line.strip()=='': continue
            line = re.sub("\s+", "__", line)
            output_texts.write(line+"\n")
            
        output_texts.close()                 

def eval_stance_result():
    '''
    evaluate results of stance detection
    global variable: output_folder
    '''
    y_true=[]# 1 pos, -1 neg, 0 neu, 2 unk (noisy data)
    y_pred=[]
    count_all_noisy, count_null, count_itag, count_null_itag = 0,0,0,0    
    count_file = 1
    set_null = set()
    set_itag = set()
    
#     eval_result_file = codecs.open('eval_result.text', 'wb') 
    for file_name in os.listdir(output_folder):
        print(file_name)

        if '.ipynb_checkpoints' in file_name or '.json' in file_name: 
            count_file+=1; continue
        file_path = output_folder + file_name
        texts = codecs.open(file_path, 'r', encoding='utf-8').read().split()#.split('\n')
        count_line = 1
        for line in texts:
            if line.strip()=='': count_line+=1; continue
            toks = line.split('__')
            print(toks)
#             if toks[1][:2] in ('I-'):#,'O-'):
#                 set_itag.add(str(count_file)+":"+str(count_line))
                
            if len(toks[1])>0 and toks[1][:2] in ('B-'):#,'I-'):#,'O-'):
                print(count_line, toks[1])
                
                if toks[3] in ('+', 'n', '-', 'null', '^n', 'g','p'):
                    y_true.append(int(toks[3].replace('+','0').replace('-','1').replace('null','2').replace('^n','2')\
                                      .replace('n','2').replace('g','2').replace('p','0')))
#                 elif toks[3] in ('null'):
#                     y_true.append(int(toks[3].replace('null','2')))
#                     set_null.add(str(count_file)+":"+str(count_line))
                    
                    if toks[-1] in ('POS', 'NEG', 'NEU'):
                        y_pred.append(int(toks[-1].replace('POS','0').replace('NEG','1').replace('NEU','2')))
    #                 else:
    #                     if toks[3] in ('null'):
    #                         y_pred.append(3)
    #                     if toks[1][:2] in ('I-'):#,'O-'):
    #                         set_itag.add(str(count_file)+":"+str(count_line))    

                assert len(y_true) == len(y_pred), "len(y_true) == len(y_pred) %d, %d, %s, %s %s"%(len(y_true), len(y_pred), file_name, count_line, toks)
                    

                
            count_line+=1
             
        assert len(y_true) == len(y_pred), "len(y_true) == len(y_pred) %d, %d"%(len(y_true), len(y_pred))
        count_file+=1    

#     eval_result_dict = metrics.classification_report(y_true, y_pred, target_names=['pos','neg','neu'], output_dict=True)
    eval_result_dict = metrics.classification_report(y_true, y_pred, output_dict=True)
#     print(result)
        
#         print('Noisy input', len([i  for i in y_pred if i==3]))

    len_intersection = len(set_null.intersection(set_itag))
    len_null = len(set_null)
    len_itag = len(set_itag)

    count_all_noisy+=len_null+len_itag-len_intersection; count_null+=len_null-len_intersection;
    count_itag+=len_itag-len_intersection; count_null_itag += len_intersection
    print('Input noisy: all (%d), only null (%d), only itag (%d), null&itag (%d)' 
          %(count_all_noisy, count_null, count_itag, count_null_itag))
    
    eval_result_dict['all_noisy'] = count_all_noisy
    eval_result_dict['only_null'] = count_null
    eval_result_dict['only_itag'] = count_itag
    eval_result_dict['only_null_itag'] = count_null_itag
    
    write_dict(eval_result_dict, output_folder+"eval_result_dict.json")
    

def eval_stance_result_check():
    '''
    evaluate results of stance detection
    global variable: output_folder
    '''
    y_true=[]# 1 pos, -1 neg, 0 neu, 2 unk (noisy data)
    y_pred=[]
    count_all_noisy, count_null, count_itag, count_null_itag = 0,0,0,0    
    count_file = 1
    set_null = set()
    set_itag = set()
    
#     eval_result_file = codecs.open('eval_result.text', 'wb') 
    for file_name in os.listdir(output_folder):
        print(file_name)

        if '.ipynb_checkpoints' in file_name or '.json' in file_name: 
            count_file+=1; continue
        file_path = output_folder + file_name
        texts = codecs.open(file_path, 'r', encoding='utf-8').read().split()#.split('\n')
        count_line = 1
        for line in texts:
            if line.strip()=='': count_line+=1; continue
            toks = line.split('__')
            print(toks)
#             if toks[1][:2] in ('I-'):#,'O-'):
#                 set_itag.add(str(count_file)+":"+str(count_line))
                
            if len(toks[1])>0 and toks[1][:2] in ('B-'):#,'I-'):#,'O-'):
                print(count_line, toks[1])
                
                if toks[3] in ('+', 'n', '-', 'null', '^n', 'g','p'):
                    y_true.append(int(toks[3].replace('+','0').replace('-','1').replace('null','2').replace('^n','2')\
                                      .replace('n','2').replace('g','2').replace('p','0')))
#                 elif toks[3] in ('null'):
#                     y_true.append(int(toks[3].replace('null','2')))
#                     set_null.add(str(count_file)+":"+str(count_line))
                    
#                     if toks[-1] in ('POS', 'NEG', 'NEU'):
                    y_pred.append(toks[-1])
    #                 else:
    #                     if toks[3] in ('null'):
    #                         y_pred.append(3)
    #                     if toks[1][:2] in ('I-'):#,'O-'):
    #                         set_itag.add(str(count_file)+":"+str(count_line))    

                assert len(y_true) == len(y_pred), "len(y_true) == len(y_pred) %d, %d, %s, %s %s"%(len(y_true), len(y_pred), file_name, count_line, toks)
                    

                
            count_line+=1
             
        assert len(y_true) == len(y_pred), "len(y_true) == len(y_pred) %d, %d"%(len(y_true), len(y_pred))
        count_file+=1    

    print(y_true)
    print(y_pred)
    
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# text =  'editing by Andrea Shalal and Adrian Croft)\n\xa9 Thomson Reuters'
# text = 'Bank pays out \xa32.5m'
# text = '\xb0C'
# print(text)   
# # text = unicode_to_ascii('editing by Andrea Shalal and Adrian Croft)\n\xa9 Thomson Reuters')
# # print(text);
# text = json.dumps(text)
# print(text) 
# exit()    
def convert2json():
    dict_file_path = "../json_data/polarity_data.js"
    text = codecs.open(dict_file_path, 'rb', encoding='utf-8').read()
#     text = text.replace("'",'"').replace('u"', '"').replace('\\"', "'")
    text= text.replace("'content'", '"content"')
    text= text.replace("'docnoId'", '"docnoId"')
    text= text.replace("'entities'", '"entities"')
    text= text.replace("'entityId'", '"entityId"')
    text= text.replace("'name'", '"name"')
    text= text.replace("'offsets'", '"offsets"')
    text= text.replace("u'end'", '"end"')
    text= text.replace("u'start'", '"start"')
    text= text.replace("'polarity'", '"polarity"')
    text= text.replace("'headline'", '"headline"')
    text= text.replace("'url'", '"url"')
        
    open(dict_file_path.replace(".js", "_.js"), 'w').write(text)
    
    lines = open(dict_file_path.replace(".js", "_.js"), 'r').readlines()
    
    wtext = open(dict_file_path.replace(".js", ".json"), 'w') 
    
    count=1
    for line in lines:

        if '{"content"' in line[:len('[{"content"')]:
#             line = line.replace(r'\\', r'\\');
            line = line.replace("u'",'"').replace('u"','"').replace("\\'", "'")           
            line = line[:-3] + '",'
#             line=line[:15]+ json.dumps(line[15: -2])[1:-1] + line[-2:]
#             print(line[15: -2])
#             print(json.dumps(line[15: -2]))            
            line=line[:15]+ line[15: -2].replace('"','\\"') + line[-2:]
#             line = unicode_to_ascii(line)
#             line = json.dumps(line)#, ensure_ascii=False).encode('utf8')
        elif '"name":' in line:
            line = line.replace("u'",'"').replace('u"','"')
            line = line[:-3] + '",'
            line.split(":")[1][1:-2]
        elif '"url":' in line:
            line = line.replace("u'",'"').replace('u"','"')
            if count==len(lines):
                line = line[:-4] + '"}]'            
            else:
                line = line[:-4] + '"},'
        else:
            line = line.rstrip().replace("u'",'"').replace("'",'"')
#         print(lines[0])
        wtext.write(line+"\n")
        count+=1
    
def convert_to_right_format(postprocess_data, dict_file_path, start_docid, end_docid):
#     postprocess_data = data_folder+"PULS_data_input/"
#     dict_file_path = data_folder+"PULS_data/polarity_data.json"    
    dict_tmp = read_dict(dict_file_path)
    dict_gt = {}
    test_data_flag = False
    for item in dict_tmp:
        content = item['content']; 
        docnoid = item['docnoId'] 
        if  docnoid==start_docid: 
            test_data_flag=True
        elif docnoid==end_docid:
            test_data_flag=False
            
        if test_data_flag: 
            dict_nepos_replace = {}    
            
    #         content = content.replace('\n', ' ')
            
            for ne in item['entities']:
                entityid = str(ne['entityId'])
                if ne['polarity']=='contradiction': polarity='c'
                elif ne['polarity']>0.5:
                    polarity = '+' 
                elif ne['polarity']==0.5:
                    polarity = 'n'
                else:
                    polarity = '-'
                for ne_pos in ne['offsets']:
                    ne_beg = ne_pos['start']; ne_end=ne_pos['end']
                    ne_txt = content[ne_beg:ne_end]
                    toks = ne_txt.split()
                    if len(toks)==0: continue
    #                 print(ne_txt)
                    ch_index = ne_beg
                    new_tok = toks[0]+"__B-x__x__" + polarity
                    dict_nepos_replace[ch_index] = new_tok
                    ch_index += len(toks[0])+1
                    for tok in toks[1:]:
                        new_tok = tok+"__I-x__x__" + polarity
                        dict_nepos_replace[ch_index] = new_tok
                        ch_index += len(tok)+1
                         
            
                char_index = 0; toks = content.split()
                new_content=[]
                for tok_index in range(len(toks)):
                    tok = toks[tok_index]
                    
                    if char_index in dict_nepos_replace:
                        new_content.append(dict_nepos_replace[char_index])
                    else:
                        new_content.append(tok+"__O")
                        
                    char_index+= len(tok)+1
    #             print(' '.join(new_content))
            
                file_name = docnoid+"_"+entityid+".txt"
                open(postprocess_data+file_name, 'w').write(' '.join(new_content))
                print(dict_gt)
    #             assert file_name not in dict_gt or file_name=='793907837DA57FD27CBF4E8A5FD3FB38_28469.txt', ("file name %s", file_name)
                dict_gt[file_name]=polarity
    write_dict(dict_gt, results_folder+'dict_gt.json')

def create_csv_BERT_PULS():
#     dict_file_path = "../json_data/polarity_data.json" 
    dict_file_path = data_folder+"PULS_data/polarity_data.json"     
    start_docid = "0A21D5765ED3B8F51065D3CF7339371E"
    end_docid =   "43B3E9B3BD827B0E9506D837408C8AB9"
    dataset_name = "PULS_data_train"
#     start_docid = "43B3E9B3BD827B0E9506D837408C8AB9" 
#     end_docid =   "7272C54D09FAFF86E978AD6072606DD6"
#     dataset_name = "PULS_data_dev"
#     start_docid = "7272C54D09FAFF86E978AD6072606DD6"
#     end_docid =   "" 
#     dataset_name = "PULS_data_test"   

    postprocess_data = data_folder+dataset_name+"_input/"
    if not os.path.exists(postprocess_data):
        os.mkdir(postprocess_data)
        
    dict_tmp = read_dict(dict_file_path)#"../json_data/polarity_data_ex.csv"
    test_data_flag=False
    with open(postprocess_data+"polarity_data.csv", mode='w') as polar_data:
        polar_writer = csv.writer(polar_data, delimiter=',', quotechar='"')
        polar_writer.writerow(['Content', 'NamedEntity', 'Polarity'])

        for item in dict_tmp:
            content = item['content']#.replace("\n", ' ')
            docnoid = item['docnoId'] 
            if  docnoid==start_docid: 
                test_data_flag=True
            elif docnoid==end_docid:
                test_data_flag=False
                
            if test_data_flag: 
                for ne in item['entities']:
                    ne_name = ne['name']
                    if ne['polarity']!='contradiction':
                        if ne['polarity']>0.5:
                            polarity = 0 
                        elif ne['polarity']==0.5:
                            polarity = 2
                        else:
                            polarity = 1
                        
                        
                        ne_beg = ne['offsets'][0]['start']
                            
                        
                        polar_writer.writerow([content[ne_beg:], ne_name, polarity])    

def create_csv_BERT_EMM():
#     data_folder="../"
    dict_file_path = data_folder+"EMM_data/EMM_data.xlsx"
    df = pandas.read_excel(dict_file_path, header=0, 
                       names=['NewsID', 'Quote', 'SourceName', 'SourceID', 'TargetID', 'TargetName', 'Ann1', 'Ann4', 'Agreement'])
    
#     start_docid = "apakistannews-71ecab5fcf1c0337d1c3e648a0eeafab"
#     end_docid =   "xinhuanet_en-e3cba5e738db06e135a075931e187a2b"
#     dataset_name = "EMM_data_train"
#     start_docid = "xinhuanet_en-e3cba5e738db06e135a075931e187a2b" 
#     end_docid =   "earthtimes-38d7131a19c247e513378cb4b13e5fd0"
#     dataset_name = "EMM_data_dev"
    start_docid = "earthtimes-38d7131a19c247e513378cb4b13e5fd0"
    end_docid =   "" 
    dataset_name = "EMM_data_test"     
    list_NewsID =  df.NewsID.values
    list_Quote = df.Quote.values
    list_TargetName = df.TargetName.values
    list_Ann1 = df.Ann1.values
    print(len(list_NewsID))
    
    test_data_flag=False
    postprocess_data = data_folder+dataset_name+"_input/"
    if not os.path.exists(postprocess_data):
        os.mkdir(postprocess_data)
            
    with open(postprocess_data+"EMM_data.csv", mode='w') as polar_data:
        polar_writer = csv.writer(polar_data, delimiter=',', quotechar='"')
        polar_writer.writerow(['Content', 'NamedEntity', 'Polarity'])    
    
        for i in range(len(list_NewsID)):
            docnoid = list_NewsID[i]
            if  docnoid==start_docid: 
                test_data_flag=True
            elif docnoid==end_docid:
                test_data_flag=False   
            if test_data_flag: 
                polarity=-1
                if list_Ann1[i]=='POS': polarity=0
                elif   list_Ann1[i]=='NEG': polarity=1
                else: polarity=2
                polar_writer.writerow([list_Quote[i], list_TargetName[i], polarity])
        
       
if __name__ == '__main__':
#*** create train, dev, test part data for BERT ***   
    create_csv_BERT_PULS()
#     create_csv_BERT_EMM()        
#     exit()

    
    
    
    
    
    
    
    
    
    
    
    



    
    
