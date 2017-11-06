import os
import sys
import string
import numpy as np
import pickle


ace_event_type_list=['none', 'elect', 'attack', 'end-position', 'sentence', 'convict', 'transfer-ownership', 'fine', 'transfer-money', 'meet', 'start-position', 'phone-write', 'start-org', 'trial-hearing', 'transport', 'die', 'arrest-jail', 'charge-indict', 'release-parole', 'demonstrate', 'execute', 'divorce', 'appeal', 'be-born', 'end-org', 'injure', 'marry', 'sue', 'declare-bankruptcy', 'nominate', 'extradite', 'acquit', 'merge-org', 'pardon']
# punc_list=['(', ')', '"', ',"', '."', '????-??-?', '"?', ').', '.,', u',''', '[', ']', '):', '?"', '),', '.)']

data_save_path="D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_data.data"
form_data_save_path = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_data_form.data"

class_size = 34
max_len = 60
sen_min_len = 5
word2vec_len = 300

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
def dealTriggerFile(file_txt,vec_dict):

    X=[]
    Y=[]
    W=[]

    trigger_file = open(file_txt)
    for line in trigger_file:
        word_list, tag_list=splitLineToTag(line.lower())

        sen_list = []
        label_list = []
        sen_word_list = []

        for i, (word, tag) in enumerate(zip(word_list, tag_list)):
            # print(i,word,tag)
            if vec_dict.get(word) is not None:
                sen_list.append(vec_dict.get(word))
                label_list.append(ace_event_type_list.index(tag))
                sen_word_list.append(word)
            # else:
            #     if word.isalnum():
            #         sen_list.append([0.0 ])
            #         label_list.append(ace_event_type_list.index(tag))
            #         sen_word_list.append(word)

        X.append(sen_list)
        Y.append(label_list)
        W.append(sen_word_list)
    return X,Y,W


def save_data():

    test_txt = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_raw/test.triggerEvent.txt"
    dev_txt = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_raw/dev.triggerEvent.txt"
    train_txt = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/trigger_raw/train.triggerEvent.txt"

    wordvec=get_word2vec()

    X_test, Y_test, W_test=dealTriggerFile(test_txt,wordvec)
    X_dev, Y_dev, W_dev=dealTriggerFile(dev_txt,wordvec)
    X_train, Y_train, W_train=dealTriggerFile(train_txt,wordvec)

    data = X_train, Y_train, W_train, X_test, Y_test, W_test, X_dev, Y_dev, W_dev
    f = open(data_save_path, 'wb')
    pickle.dump(data, f)

    print(np.array(X_train).shape)
    print(np.array(Y_train).shape)
    print(np.array(W_train).shape)
    print(np.array(X_test).shape)
    print(np.array(Y_test).shape)
    print(np.array(W_test).shape)
    print(np.array(X_dev).shape)
    print(np.array(Y_dev).shape)
    print(np.array(W_dev).shape)



# 规范句子长度
def padding_mask(x, y, w, max_len):
    X_train = []    # 词向量
    Y_train = []    # 标签
    L_train=[]      # 序列长度
    Weights_train=[] # 标签权重
    W_train = []    # 单词
    x_zero_list = [0.0 for i in range(word2vec_len)]
    # y_zero_list = [0.0 for i in range(class_size)]
    # y_zero_list[0] = 1.0
    unknown = '#'
    for i, (x, y, w) in enumerate(zip(x, y, w)):
        if max_len > len(x):
            L_train.append(len(x))
            for j in range(max_len - len(x)):
                x.append(x_zero_list)
                y.append(0)
                w.append(unknown)
        else:
            L_train.append(max_len)
            x = x[:max_len]
            y = y[:max_len]
            w = w[:max_len]
        X_train.append(x)
        Y_train.append(y)
        W_train.append(w)

    for i in range(len(L_train)):
        tag_weights_row=[]
        for j in range(max_len):
            if j<=L_train[i]:
                tag_weights_row.append(1)
            else:
                tag_weights_row.append(0)
        Weights_train.append(tag_weights_row)

    # tag=[]
    # for i in range(len(Y_train)):
    #     tag_row=[]
    #     for j in range(len(Y_train[i])):
    #         label_value=Y_train[i][j].index(1.0)
    #         if label_value!=0:
    #             print(label_value)
    #         tag_row.append(label_value)
    #     tag.append(tag_row)

    return X_train, Y_train, L_train, Weights_train, W_train


def form_data():
    data_f = open(data_save_path, 'rb')
    X_train, Y_train, W_train, X_test, Y_test, W_test, X_dev, Y_dev, W_dev = pickle.load(data_f)
    data_f.close()

    X_train, tag_train, L_train, Weights_train, W_train = padding_mask(X_train, Y_train, W_train, max_len)
    X_test, tag_test, L_test, Weights_test, W_test = padding_mask(X_test, Y_test, W_test, max_len)
    X_dev, tag_dev, L_dev, Weights_dev, W_dev = padding_mask(X_dev, Y_dev, W_dev, max_len)

    data = X_train, tag_train, L_train, Weights_train, W_train, X_test, tag_test, L_test, Weights_test, W_test, X_dev, tag_dev, L_dev, Weights_dev, W_dev
    f = open(form_data_save_path, 'wb')
    pickle.dump(data, f)

    print(np.array(X_train).shape)
    print(np.array(tag_train).shape)
    print(np.array(L_train).shape)
    print(np.array(Weights_train).shape)
    print(np.array(W_train).shape)

    print(np.array(X_test).shape)
    print(np.array(tag_test).shape)
    print(np.array(L_test).shape)
    print(np.array(Weights_test).shape)
    print(np.array(W_test).shape)

    print(np.array(X_dev).shape)
    print(np.array(tag_dev).shape)
    print(np.array(L_dev).shape)
    print(np.array(Weights_dev).shape)
    print(np.array(W_dev).shape)


if __name__ == "__main__":

    save_data()

    form_data()



# (14227, 60, 300)
# (14227, 60)
# (14227,)
# (14227, 60)
# (14227, 60)
# (739, 60, 300)
# (739, 60)
# (739,)
# (739, 60)
# (739, 60)
# (2301, 60, 300)
# (2301, 60)
# (2301,)
# (2301, 60)
# (2301, 60)



