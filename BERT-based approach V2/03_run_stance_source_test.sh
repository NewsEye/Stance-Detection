#!/bin/bash

fi_cased=3
fi_uncased=9

fr_cased=0
fr_uncased=7

ge_cased=2
ge_uncased=1

sv_cased=9
sv_uncased=9

combine_str=$fi_cased,$fi_uncased,$fr_cased,$fr_uncased,$ge_cased,$ge_uncased,$sv_cased,$sv_uncased

echo $combine_str

python BERT_Stance_test_newseye.py Finnish 1 128 $fi_cased

# python BERT_Stance_test_newseye.py Finnish 0 128 $fi_uncased

python BERT_Stance_test_newseye.py French 1 128 $fr_cased

# python BERT_Stance_test_newseye.py French 0 128 $fr_uncased

python BERT_Stance_test_newseye.py German 1 128 $ge_cased

# python BERT_Stance_test_newseye.py German 0 128 $ge_uncased

python BERT_Stance_test_newseye.py Swedish 1 128 $sv_cased

# python BERT_Stance_test_newseye.py Swedish 0 128 $sv_uncased

python utils.py 'BERT' $combine_str

