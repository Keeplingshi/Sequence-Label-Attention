# -*- coding: utf-8 -*-
"""

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import logging

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# import main.data_utils as data_utils
import main.sequence_model as sequence_model
import argparse,pickle

import subprocess
import stat


def test():
    # data_f = open("D:/Code/pycharm/Sequence-Label-Attention/main/saver/argument_saver/temp_data.data", 'rb')
    # pred_test, tag_test, L_test = pickle.load(data_f)
    # data_f.close()

    data_f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/argument_raw/argument_data_form.data", 'rb')
    X_train, tag_train, L_train, Weights_train, W_train, X_test, tag_test, L_test, Weights_test, W_test, X_dev, tag_dev, L_dev, Weights_dev, W_dev = pickle.load(data_f)
    data_f.close()

    # 触发词类型
    T_test=[]
    X_test=np.array(X_test)
    X_test=X_test[:,:,305:]
    for i in X_test:
        for j in i:
            if 1.0 in j:
                T_test.append(j.tolist().index(1.0))
                break


    T_train=[]
    X_train=np.array(X_train)
    X_train=X_train[:,:,305:]
    for i in X_train:
        for j in i:
            if 1.0 in j:
                T_train.append(j.tolist().index(1.0))
                break


    T_dev=[]
    X_dev=np.array(X_dev)
    X_dev=X_dev[:,:,305:]
    # print(np.array(X_dev).shape)
    for i in X_dev:
        for j in i:
            if 1.0 in j:
                T_dev.append(j.tolist().index(1.0))
                break



    ACE_EVENT_Trigger_Type=[None, 'Trigger_Transport', 'Trigger_Elect', 'Trigger_Start-Position', 'Trigger_Nominate', 'Trigger_Attack', 'Trigger_End-Position', 'Trigger_Meet', 'Trigger_Marry', 'Trigger_Phone-Write', 'Trigger_Transfer-Money', 'Trigger_Sue', 'Trigger_Demonstrate', 'Trigger_End-Org', 'Trigger_Injure', 'Trigger_Die', 'Trigger_Arrest-Jail', 'Trigger_Transfer-Ownership', 'Trigger_Start-Org', 'Trigger_Execute', 'Trigger_Trial-Hearing', 'Trigger_Sentence', 'Trigger_Be-Born', 'Trigger_Charge-Indict', 'Trigger_Convict', 'Trigger_Declare-Bankruptcy', 'Trigger_Release-Parole', 'Trigger_Fine', 'Trigger_Pardon', 'Trigger_Appeal', 'Trigger_Merge-Org', 'Trigger_Extradite', 'Trigger_Divorce', 'Trigger_Acquit']
    ACE_EVENT_Argument_Type=[None, 'Argument_Vehicle', 'Argument_Artifact', 'Argument_Destination', 'Argument_Agent', 'Argument_Person', 'Argument_Position', 'Argument_Entity', 'Argument_Attacker', 'Argument_Place', 'Argument_Time-At-Beginning', 'Argument_Target', 'Argument_Giver', 'Argument_Recipient', 'Argument_Plaintiff', 'Argument_Money', 'Argument_Victim', 'Argument_Time-Within', 'Argument_Buyer', 'Argument_Time-Ending', 'Argument_Instrument', 'Argument_Seller', 'Argument_Origin', 'Argument_Time-Holds', 'Argument_Org', 'Argument_Time-At-End', 'Argument_Time-Before', 'Argument_Time-Starting', 'Argument_Time-After', 'Argument_Beneficiary', 'Argument_Defendant', 'Argument_Adjudicator', 'Argument_Sentence', 'Argument_Crime', 'Argument_Prosecutor', 'Argument_Price']


    event_template=dict()
    for i in range(1,34):
        event_template[i]=[]

    for i,j in zip(T_train,tag_train):
        for tmp in j:
            if tmp!=0 and tmp not in event_template[i]:
                event_template[i].append(tmp)


    for i,j in zip(T_dev,tag_dev):
        for tmp in j:
            if tmp!=0 and tmp not in event_template[i]:
                event_template[i].append(tmp)


    for i,j in zip(T_test,tag_test):
        for tmp in j:
            if tmp!=0 and tmp not in event_template[i]:
                event_template[i].append(tmp)

    # print(event_template)
    event_template_a=dict()
    for (k,v) in  event_template.items():
        event_template_a[ACE_EVENT_Trigger_Type[k].replace("Trigger_","").lower()]=[ACE_EVENT_Argument_Type[t].replace("Argument_","").lower() for t in v]

    return event_template_a


def get_event_template():
    f = open("D:/Code/pycharm/Sequence-Label-Attention/data/ACE-event-argument.txt")             # 返回一个文件对象
    line = f.readline()             # 调用文件的 readline()方法
    event_template_b=dict()
    while line:
        a=line.replace("\n","").split(":")
        trigger=a[0].lower()
        s=a[1].lower().split(" ")[1:]
        event_template_b[trigger]=s
        line = f.readline()

    f.close()
    return event_template_b


def save_event_template():
    ACE_EVENT_Trigger_Type=['Trigger_None', 'Trigger_Transport', 'Trigger_Elect', 'Trigger_Start-Position', 'Trigger_Nominate', 'Trigger_Attack', 'Trigger_End-Position', 'Trigger_Meet', 'Trigger_Marry', 'Trigger_Phone-Write', 'Trigger_Transfer-Money', 'Trigger_Sue', 'Trigger_Demonstrate', 'Trigger_End-Org', 'Trigger_Injure', 'Trigger_Die', 'Trigger_Arrest-Jail', 'Trigger_Transfer-Ownership', 'Trigger_Start-Org', 'Trigger_Execute', 'Trigger_Trial-Hearing', 'Trigger_Sentence', 'Trigger_Be-Born', 'Trigger_Charge-Indict', 'Trigger_Convict', 'Trigger_Declare-Bankruptcy', 'Trigger_Release-Parole', 'Trigger_Fine', 'Trigger_Pardon', 'Trigger_Appeal', 'Trigger_Merge-Org', 'Trigger_Extradite', 'Trigger_Divorce', 'Trigger_Acquit']
    ACE_EVENT_Argument_Type=['Argument_None', 'Argument_Vehicle', 'Argument_Artifact', 'Argument_Destination', 'Argument_Agent', 'Argument_Person', 'Argument_Position', 'Argument_Entity', 'Argument_Attacker', 'Argument_Place', 'Argument_Time-At-Beginning', 'Argument_Target', 'Argument_Giver', 'Argument_Recipient', 'Argument_Plaintiff', 'Argument_Money', 'Argument_Victim', 'Argument_Time-Within', 'Argument_Buyer', 'Argument_Time-Ending', 'Argument_Instrument', 'Argument_Seller', 'Argument_Origin', 'Argument_Time-Holds', 'Argument_Org', 'Argument_Time-At-End', 'Argument_Time-Before', 'Argument_Time-Starting', 'Argument_Time-After', 'Argument_Beneficiary', 'Argument_Defendant', 'Argument_Adjudicator', 'Argument_Sentence', 'Argument_Crime', 'Argument_Prosecutor', 'Argument_Price']

    ACE_EVENT_Trigger_Type=[i.replace("Trigger_","").lower() for i in ACE_EVENT_Trigger_Type]
    ACE_EVENT_Argument_Type=[i.replace("Argument_","").lower() for i in ACE_EVENT_Argument_Type]

    time_list=[]
    for i in ACE_EVENT_Argument_Type:
        if "time" in i:
            time_list.append(i)

    event_template_b=get_event_template()

    event_template_num_dict=dict()
    for k,v in event_template_b.items():
        if "time" in v:
            v.remove("time")
            v.extend(time_list)
        argument_num_list=[ACE_EVENT_Argument_Type.index(i) for i in v]
        event_template_num_dict[ACE_EVENT_Trigger_Type.index(k)]=argument_num_list


    f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/event_template.data", 'wb')
    pickle.dump(event_template_num_dict, f)


def argument_calculate_f_score():

    data_f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/temp_data.data", 'rb')
    pred_test, tag_test, L_test, T_test = pickle.load(data_f)
    data_f.close()

    template_f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/event_template.data", 'rb')
    event_template_num_dict = pickle.load(template_f)
    template_f.close()

    # print(pred_test)
    # print(tag_test)
    print(np.array(pred_test).shape)        #预测结果
    print(np.array(tag_test).shape)         #真实结果
    print(np.array(T_test).shape)           #事件类型

    # prediction = np.argmax(pred_test, 2)
    # print(np.array(prediction).shape)
    prediction=[]

    for i,(event_type,pred_test) in enumerate(zip(T_test,pred_test)):
        print(i,event_type,pred_test)
        event_template=event_template_num_dict[event_type]
        event_template.append(0)
        print(np.array(pred_test).shape)
        sen_pred=[]
        for pred_type in pred_test:
            max_index=get_max_from_list(pred_type,event_template)
            sen_pred.append(max_index)

        prediction.append(sen_pred)


    print(np.array(prediction).shape)
    #========================计算F值-===========================
    iden_p=0   # 识别的个体总数
    iden_r=0    # 测试集中存在个个体总数
    iden_acc=0  # 正确识别的个数

    classify_p = 0  # 识别的个体总数
    classify_r = 0  # 测试集中存在个个体总数
    classify_acc = 0  # 正确识别的个数

    for i in range(len(L_test)):
        for j in range(L_test[i]):
            if prediction[i][j]!=0:
                classify_p+=1
                iden_p+=1

            if tag_test[i][j]!=0:
                classify_r+=1
                iden_r+=1

            if tag_test[i][j]==prediction[i][j] and tag_test[i][j]!=0:
                classify_acc+=1

            if prediction[i][j]!=0 and tag_test[i][j]!=0:
                iden_acc+=1

    try:
        print('\n\nArgument Identification:')
        print(str(iden_acc) + '------' + str(iden_p) + '------' + str(iden_r))
        p = iden_acc / iden_p
        r = iden_acc / iden_r
        if p + r != 0:
            f = 2 * p * r / (p + r)
            print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
        print('Argument Classification:')
        print(str(classify_acc) + '------' + str(classify_p) + '------' + str(classify_r))
        p = classify_acc / classify_p
        r = classify_acc / classify_r
        if p + r != 0:
            f = 2 * p * r / (p + r)
            print('P=' + str(p) + "\tR=" + str(r) + "\tF=" + str(f))
    except ZeroDivisionError:
        print('all zero')


def get_max_from_list(pred_type,event_template):

    max_index=-1
    max_value=float('-Inf')
    for i in event_template:
        if pred_type[i]>max_value:
            max_value=pred_type[i]
            max_index=i

    return max_index


# Argument Identification:
# 678------1166------1249
# P=0.58147512864494	R=0.5428342674139311	F=0.5614906832298135
# Argument Classification:
# 549------1166------1249
# P=0.47084048027444253	R=0.43955164131305047	F=0.4546583850931677

if __name__ == "__main__":

    # pred_type=[9,2,5,7,4]
    # event_template=[1,2,3]
    # get_max_from_list(pred_type,event_template)

    # save_event_template()

    argument_calculate_f_score()

    # ACE_EVENT_Trigger_Type=['Trigger_None', 'Trigger_Transport', 'Trigger_Elect', 'Trigger_Start-Position', 'Trigger_Nominate', 'Trigger_Attack', 'Trigger_End-Position', 'Trigger_Meet', 'Trigger_Marry', 'Trigger_Phone-Write', 'Trigger_Transfer-Money', 'Trigger_Sue', 'Trigger_Demonstrate', 'Trigger_End-Org', 'Trigger_Injure', 'Trigger_Die', 'Trigger_Arrest-Jail', 'Trigger_Transfer-Ownership', 'Trigger_Start-Org', 'Trigger_Execute', 'Trigger_Trial-Hearing', 'Trigger_Sentence', 'Trigger_Be-Born', 'Trigger_Charge-Indict', 'Trigger_Convict', 'Trigger_Declare-Bankruptcy', 'Trigger_Release-Parole', 'Trigger_Fine', 'Trigger_Pardon', 'Trigger_Appeal', 'Trigger_Merge-Org', 'Trigger_Extradite', 'Trigger_Divorce', 'Trigger_Acquit']
    # ACE_EVENT_Argument_Type=['Argument_None', 'Argument_Vehicle', 'Argument_Artifact', 'Argument_Destination', 'Argument_Agent', 'Argument_Person', 'Argument_Position', 'Argument_Entity', 'Argument_Attacker', 'Argument_Place', 'Argument_Time-At-Beginning', 'Argument_Target', 'Argument_Giver', 'Argument_Recipient', 'Argument_Plaintiff', 'Argument_Money', 'Argument_Victim', 'Argument_Time-Within', 'Argument_Buyer', 'Argument_Time-Ending', 'Argument_Instrument', 'Argument_Seller', 'Argument_Origin', 'Argument_Time-Holds', 'Argument_Org', 'Argument_Time-At-End', 'Argument_Time-Before', 'Argument_Time-Starting', 'Argument_Time-After', 'Argument_Beneficiary', 'Argument_Defendant', 'Argument_Adjudicator', 'Argument_Sentence', 'Argument_Crime', 'Argument_Prosecutor', 'Argument_Price']
    #
    # ACE_EVENT_Trigger_Type=[i.replace("Trigger_","").lower() for i in ACE_EVENT_Trigger_Type]
    # print(ACE_EVENT_Trigger_Type)
    # ACE_EVENT_Argument_Type=[i.replace("Argument_","").lower() for i in ACE_EVENT_Argument_Type]
    # print(ACE_EVENT_Argument_Type)
    #
    # time_list=[]
    # for i in ACE_EVENT_Argument_Type:
    #     if "time" in i:
    #         time_list.append(i)
    #         print(i)
    #     # print(i)
    # print(time_list)
    # # sys.exit()
    #
    # # event_template_a=argument_calculate_f_score()
    # event_template_b=get_event_template()
    #
    # # print(event_template_a.keys())
    # # print(event_template_b.keys())
    #
    # event_template_num_dict=dict()
    # for k,v in event_template_b.items():
    #     # print(k,v)
    #     # print(ACE_EVENT_Trigger_Type.index(k))
    #     if "time" in v:
    #         v.remove("time")
    #         v.extend(time_list)
    #     argument_num_list=[ACE_EVENT_Argument_Type.index(i) for i in v]
    #     # print(argument_num_list)
    #     event_template_num_dict[ACE_EVENT_Trigger_Type.index(k)]=argument_num_list
    #
    # f = open("D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/event_template.data", 'wb')
    # pickle.dump(event_template_num_dict, f)

    # for trigger in event_template_a.keys():
    #     if trigger not in event_template_b.keys():
    #         print("12222222222222222")
    #     # print(event_template_a[trigger])
    #     # print(event_template_b[trigger])
    #
    #     for i in event_template_a[trigger]:
    #         # 判断在a中，却不在b中的
    #         # print(i)
    #         if "time" not in i and i not in event_template_b[trigger]:
    #             print(trigger)
    #             print(i)
    #             print(event_template_a[trigger])
    #             print(event_template_b[trigger])
    #             print("================================")



# nominate
# agent
# ['person', 'agent', 'time-within', 'position']
# ['person', 'entity', 'position', 'time', 'place']
# ================================
# appeal
# plaintiff
# ['plaintiff', 'adjudicator', 'place', 'crime', 'time-within', 'time-holds']
# ['defendant', 'prosecutor', 'adjudicator', 'crime', 'time', 'place']
# ================================
# transport
# victim
# ['vehicle', 'artifact', 'destination', 'agent', 'time-at-beginning', 'time-within', 'origin', 'time-at-end', 'time-starting', 'time-after', 'time-holds', 'victim', 'place', 'time-ending', 'time-before']
# ['agent', 'artifact', 'vehicle', 'price', 'origin', 'destination', 'time']
# ================================
# transport
# place
# ['vehicle', 'artifact', 'destination', 'agent', 'time-at-beginning', 'time-within', 'origin', 'time-at-end', 'time-starting', 'time-after', 'time-holds', 'victim', 'place', 'time-ending', 'time-before']
# ['agent', 'artifact', 'vehicle', 'price', 'origin', 'destination', 'time']
# ================================
# die
# person
# ['victim', 'agent', 'place', 'time-within', 'instrument', 'time-before', 'time-after', 'person', 'time-starting', 'time-holds', 'time-ending', 'time-at-beginning']
# ['agent', 'victim', 'instrument', 'time', 'place']
# ================================
# attack
# agent
# ['attacker', 'place', 'target', 'time-within', 'time-ending', 'instrument', 'time-holds', 'time-at-beginning', 'time-after', 'time-starting', 'time-before', 'time-at-end', 'agent', 'victim']
# ['attacker', 'target', 'instrument', 'time', 'place']
# ================================
# attack
# victim
# ['attacker', 'place', 'target', 'time-within', 'time-ending', 'instrument', 'time-holds', 'time-at-beginning', 'time-after', 'time-starting', 'time-before', 'time-at-end', 'agent', 'victim']
# ['attacker', 'target', 'instrument', 'time', 'place']
# ================================
# phone-write
# place
# ['entity', 'time-within', 'time-before', 'place', 'time-holds', 'time-starting', 'time-after']
# ['entity', 'time']
# ================================
