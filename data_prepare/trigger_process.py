import os
import sys

def get_word2vec():
    word2vec_file="D:/Code/pycharm/Sequence-Label-Attention/data/word2vec/wordvector"
    wordlist_file="D:/Code/pycharm/Sequence-Label-Attention/data/word2vec/wordlist"

    wordvec={}
    word2vec_f=open(word2vec_file,'r')
    wordlist_f=open(wordlist_file,'r')
    word_len=19488
    for line in range(word_len):
        word=wordlist_f.readline().strip()
        vec=word2vec_f.readline().strip()
        temp=vec.split(',')
        temp = map(float, temp)
        vec_list = []
        for i in temp:
            vec_list.append(i)
        wordvec[word]=vec_list
    return wordvec

"""
切分成句子和标签
"""
def splitLineToTag(line):
    sen_list=line.split()
    list_mid=int(len(sen_list) / 2)
    word_list=sen_list[:list_mid]
    tag_list=sen_list[list_mid:]
    assert len(word_list)==len(tag_list),"句子长度与标签长度不相等"
    return word_list,tag_list


"""
处理触发词文件
"""
def dealTriggerFile(file_txt):

    X=[]
    Y=[]
    W=[]

    trigger_file = open(file_txt)
    for line in trigger_file:
        word_list, tag_list=splitLineToTag(line.lower())
        for i, (word, tag) in enumerate(zip(word_list, tag_list)):
            print(i,word,tag)
        # for word,tag in word_list,tag_list:
        #     print(word)
        #     print(tag)


if __name__ == "__main__":
    test_txt = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_raw/test.triggerEvent.txt"
    dev_txt = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_raw/dev.triggerEvent.txt"
    train_txt = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_raw/train.triggerEvent.txt"

    dealTriggerFile(test_txt)

