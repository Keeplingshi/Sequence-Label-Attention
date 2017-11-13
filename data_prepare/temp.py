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


def argument_calculate_f_score():
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


if __name__ == "__main__":

    event_template_a=argument_calculate_f_score()
    event_template_b=get_event_template()

    print(event_template_a.keys())
    print(event_template_b.keys())

    for trigger in event_template_a.keys():
        if trigger not in event_template_b.keys():
            print("12222222222222222")
        # print(event_template_a[trigger])
        # print(event_template_b[trigger])

        for i in event_template_a[trigger]:
            # 判断在a中，却不在b中的
            # print(i)
            if "time" not in i and i not in event_template_b[trigger]:
                print(trigger)
                print(i)
                print(event_template_a[trigger])
                print(event_template_b[trigger])
                print("================================")



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
