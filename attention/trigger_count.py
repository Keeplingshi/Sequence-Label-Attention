# -*- coding: utf-8 -*-

"""
统计单词的注意力影响
"""

import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl


"""
画图
"""
def draw_attention_pic(sen,tag_sen,aw_sen):
    # print(sen)
    # print(tag_sen)
    # print(aw_sen)
    # print(len(sen))

    for i in range(len(tag_sen)):
        if tag_sen[i]!=0:
            draw_bar(sen,aw_sen[i][:len(sen)],sen[i])
            #横轴x显示sen，
            # print(aw_sen[i][:len(sen)])
            # print(len(aw_sen[i][:len(sen)]))
            # plt.bar(range(len(aw_sen[i][:len(sen)])), aw_sen[i][:len(sen)])
            # plt.xlabel(sen)
            # # plt.xticks(x, names, rotation=45)
            # plt.show()




def draw_bar(labels,quants,title):
    width = 0.4
    ind = np.linspace(0.5,9.5,len(labels))
    # make a square figure
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)
    # Bar Plot
    ax.bar(ind-width/2,quants,width,color='green')

    # Set the ticks on x-axis
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    # labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # title
    ax.set_title(title, bbox={'facecolor':'0.8', 'pad':5})
    plt.grid(True)
    plt.show()
    # plt.savefig("bar.jpg")
    plt.close()


if __name__ == "__main__":

    # labels=['USA', 'China', 'India', 'Japan', 'Germany', 'Russia', 'Brazil', 'UK', 'France', 'Italy']
    #
    # quants=[15094025.0, 11299967.0, 4457784.0, 4440376.0, 3099080.0, 2383402.0, 2293954.0, 2260803.0, 2217900.0, 1846950.0]
    #
    # draw_bar(labels,quants)

    # sys.exit()


    data_save_path="D:/Code/pycharm/Sequence-Label-Attention/attention/count_data/trigger_count.data"
    data_f = open(data_save_path, 'rb')
    pred,tag,attention_weights,Len,words = pickle.load(data_f)
    pred = np.argmax(pred, 2)
    data_f.close()

    print(np.array(pred).shape)
    print(np.array(tag).shape)
    print(np.array(attention_weights).shape)
    print(np.array(Len).shape)
    print(np.array(words).shape)

    for i in range(739):

        isEqual=True
        for j in range(Len[i]):
            if pred[i][j]!=tag[i][j]:
                isEqual=False

        isTrigger=False
        if isEqual:
            for j in range(Len[i]):
                if pred[i][j]!=0:
                    isTrigger=True

        if isEqual and isTrigger and Len[i]<13:
            draw_attention_pic(words[i][:Len[i]],tag[i][:Len[i]],attention_weights[i][:Len[i]])
            # sys.exit()

