from main.data_process.xml_parse import xml_parse_base
from xml.dom import minidom

import os
import re
import string
#import gensim
#from gensim.models import word2vec
import pickle
import sys
import os
import string
import time
import numpy as np
import re


# homepath = 'D:/Code/pycharm/Sequence-Label-Attention/main/data/'
sourcepath = "D:/Code/pycharm/Sequence-Label-Attention/source/ace05/"
acepath=sourcepath+"/data/English/"

doclist_train = sourcepath + "split1.0/ace_train_filelist.txt"
doclist_test = sourcepath + "split1.0/ace_test_filelist.txt"
doclist_dev = sourcepath + "split1.0/ace_dev_filelist.txt"

data_save_path = "D:/Code/pycharm/Sequence-Label-Attention/main/data//trigger_train_data.data"
form_data_save_path = "D:/Code/pycharm/Sequence-Label-Attention/main/data//trigger_data_form.data"

word2vec_file = 'D:/Code/pycharm/Sequence-Label-Attention/source/ace05/word2vec/wordvector'
wordlist_file = 'D:/Code/pycharm/Sequence-Label-Attention/source/ace05/word2vec/wordlist'


# 定义存储ACE事件的类
class ACE_info:
    # go gain the event mention from ACE dataset
    def __init__(self):
        self.id = None                       # 获取事件编号
        self.text = None                     # 获取整个事件的内容
        self.text_start=None                 # 获取候选事件的起始位置
        self.text_end=None                   # 获取候选事件的终止位置
        self.trigger = None                  # 获取事件触发词
        self.trigger_start = None            # 获取事件触发词的起始位置
        self.trigger_end = None              # 获取事件触发词的终止位置
        self.trigger_sub_type = None         # 获取事件触发词的子类型
        self.argument = []                   # 获取事件元素
        self.argument_start = []             # 获取事件元素的起始位置
        self.argument_end = []               # 获取事件元素的终止位置
        self.argument_type = []              # 获取事件元素的类型
        self.argument_entity_type = []
        self.entity = []                     # 获取实体
        self.entity_start = []               # 获取实体的起始位置
        self.entity_end = []                 # 获取实体的终止位置
        self.entity_type = []                # 获取实体的类型

    def toString(self):
        return 'id:' + str(self.id) + '\t text:' + str(self.text) + '\t trigger:' + str(
            self.trigger) + '\t trigger_sub_type:' + str(self.trigger_sub_type)+ \
               '\t argument:' + str(self.argument) + '\t argument_type:' + str(self.argument_type)\
               + '\t trigger_start:' + str(self.trigger_start)+ '\t trigger_end:' + str(self.trigger_end) \
               + '\t argument_start:' + str(self.argument_start)+ '\t argument_end:' + str(self.argument_end)



# 定义抽取ACE事件的函数
def extract_ace_info(apf_file):
    # 存储事件实体的list
    R = []
    doc = minidom.parse(apf_file)#从xml文件得到doc对象
    root = doc.documentElement#获得根对象source_file
    entity = {}
    # 获取实体提及
    entity_nodes = xml_parse_base.get_xmlnode(None, root, 'entity')
    for entity_node in entity_nodes:
        entity_type = xml_parse_base.get_attrvalue(None, entity_node, 'SUBTYPE')
        entity_mention_nodes = xml_parse_base.get_xmlnode(None, entity_node, 'entity_mention')
        for entity_mention_node in entity_mention_nodes:
            entity_mention_id = xml_parse_base.get_attrvalue(None, entity_mention_node, 'ID')
            entity_mention_head = xml_parse_base.get_xmlnode(None, entity_mention_node, 'head')
            entity_mention_head_charseq = xml_parse_base.get_xmlnode(None, entity_mention_head[0], 'charseq')
            for charse in entity_mention_head_charseq:
                entity_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
                entity_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
                entity[entity_mention_id] = [entity_mention_start, entity_mention_end, entity_type]
    #获得value提及
    value_nodes = xml_parse_base.get_xmlnode(None, root, 'value')
    for value_node in value_nodes:
        value_type = xml_parse_base.get_attrvalue(None, value_node, 'TYPE')
        value_mention_nodes = xml_parse_base.get_xmlnode(None, value_node, 'value_mention')
        for value_mention_node in value_mention_nodes:
            value_mention_id = xml_parse_base.get_attrvalue(None, value_mention_node, 'ID')
            value_mention_extent = xml_parse_base.get_xmlnode(None, value_mention_node, 'extent')
            value_mention_extent_charseq = xml_parse_base.get_xmlnode(None, value_mention_extent[0], 'charseq')
            for charse in value_mention_extent_charseq:
                value_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
                value_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
                entity[value_mention_id] = [value_mention_start,value_mention_end,value_type]
    #获得time提及
    timex2_nodes = xml_parse_base.get_xmlnode(None, root, 'timex2')
    for timex2_node in timex2_nodes:
        timex2_mention_nodes = xml_parse_base.get_xmlnode(None, timex2_node, 'timex2_mention')
        for timex2_mention_node in timex2_mention_nodes:
            timex2_mention_id = xml_parse_base.get_attrvalue(None, timex2_mention_node, 'ID')
            timex2_mention_extent = xml_parse_base.get_xmlnode(None, timex2_mention_node, 'extent')
            timex2_mention_extent_charseq = xml_parse_base.get_xmlnode(None, timex2_mention_extent[0], 'charseq')
            for charse in timex2_mention_extent_charseq:
                timex2_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
                timex2_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
                entity[timex2_mention_id] = [timex2_mention_start,timex2_mention_end,'timex2']


    event_nodes = xml_parse_base.get_xmlnode(None, root, 'event')#获得对象名为event的节点列表
    for node in event_nodes:
        # 获取事件mention
        mention_nodes = xml_parse_base.get_xmlnode(None, node, 'event_mention')#获得对象名为event_mention的节点列表
        for mention_node in mention_nodes:
            R_element = ACE_info()
            # 获取事件id
            R_element.id = xml_parse_base.get_attrvalue(None, mention_node, 'ID')#获得event_mention的ID属性值
            # 获取事件子类型
            R_element.trigger_sub_type = xml_parse_base.get_attrvalue(None, node, 'SUBTYPE')#获得event的SUBTYPE属性值
            # 获取事件所在语句
            mention_ldc_scope = xml_parse_base.get_xmlnode(None, mention_node, 'ldc_scope')#获得ldc_scope列表
            mention_ldc_scope_charseq = xml_parse_base.get_xmlnode(None, mention_ldc_scope[0], 'charseq')
            #获得事件语句并将/n换位空格
            text_str = xml_parse_base.get_nodevalue(None, mention_ldc_scope_charseq[0], 0).replace("\n", " ")
            R_element.text = text_str
            s = None
            m = 0
            #text开始均为0
            for charse in mention_ldc_scope_charseq:
                start = xml_parse_base.get_attrvalue(None,charse, 'START')#charseq的START属性值
                end = xml_parse_base.get_attrvalue(None, charse, 'END')
                s = start
                R_element.text_start = 0
                R_element.text_end = int(end)-int(start)
                for j, x in enumerate(entity):
                    #print entity[x][0]
                    if int(entity[x][0])>=int(start) and int(entity[x][1])<=int(end):
                        R_element.entity_start.append(int(entity[x][0]) - int(s))
                        #print R_element.entity_start
                        R_element.entity_end.append(int(entity[x][1]) - int(s))
                        R_element.entity_type.append(entity[x][2])
                        R_element.entity.append(R_element.text[R_element.entity_start[m]:R_element.entity_end[m]+1])
                        m+=1


            # 获取事件触发词
            mention_anchor = xml_parse_base.get_xmlnode(None, mention_node, 'anchor')#获得anchor列表
            mention_anchor_charseq = xml_parse_base.get_xmlnode(None, mention_anchor[0], 'charseq')
            for anch in mention_anchor_charseq:
                start = xml_parse_base.get_attrvalue(None, anch, 'START')
                end = xml_parse_base.get_attrvalue(None, anch, 'END')
                R_element.trigger_start = int(start)-int(s)#相对语句位置
                R_element.trigger_end = int(end)-int(s)
                R_element.trigger = R_element.text[R_element.trigger_start:R_element.trigger_end+1]#begin 这样的



            # 获取事件元素
            mention_arguments = xml_parse_base.get_xmlnode(None, mention_node, 'event_mention_argument')#event_mention_argument列表
            i = 0
            arg = []
            for mention_argument in mention_arguments:
                mention_argument_refid = xml_parse_base.get_attrvalue(None,mention_argument, 'REFID')
                #mention_argument_extent = get_xmlnode(None, mention_argument, 'extent')
                #mention_argument_charseq = get_xmlnode(None, mention_argument_extent[0], 'charseq')
                try:
                    argument_position = entity[mention_argument_refid]
                    start = argument_position[0]
                    end = argument_position[1]
                    entity_type = argument_position[2]
                except KeyError:
                    print ('error')
                    mention_argument_extent = xml_parse_base.get_xmlnode(None, mention_argument, 'extent')
                    mention_argument_charseq = xml_parse_base.get_xmlnode(None, mention_argument_extent[0], 'charseq')
                    for argument_charse in mention_argument_charseq:
                        start = xml_parse_base.get_attrvalue(None,argument_charse, 'START')
                        end = xml_parse_base.get_attrvalue(None, argument_charse, 'END')
                        entity_type = None

                R_element.argument_start.append(int(start) - int(s))#多个事件元素
                R_element.argument_end.append( int(end) - int(s))
                R_element.argument.append(R_element.text[R_element.argument_start[i]:R_element.argument_end[i]+1])
                #arg.append(get_nodevalue(None, mention_argument_charseq[0], 0).replace("\n", " "))#事件元素列表
                R_element.argument_entity_type.append(entity_type)
                R_element.argument_type.append(xml_parse_base.get_attrvalue(None, mention_argument, 'ROLE'))#事件元素类型
                i+=1

            R.append(R_element)
    return R


def encode_corpus(flag):
    if flag == 'train':
        doclist_train_f = [acepath + i.replace('\n', '') for i in open(doclist_train, 'r')]
        ace_list = []
        for file_path in doclist_train_f:
            ace_info_list = extract_ace_info(file_path+ ".apf.xml")
            ace_list.extend(ace_info_list)  #追加一个列表
        return ace_list
    if flag == 'test':
        doclist_train_f = [acepath + i.replace('\n', '') for i in open(doclist_test, 'r')]
        ace_list = []
        for file_path in doclist_train_f:
            ace_info_list = extract_ace_info(file_path+ ".apf.xml")
            ace_list.extend(ace_info_list)  #追加一个列表
        return ace_list
    if flag == 'dev':
        doclist_train_f = [acepath + i.replace('\n', '') for i in open(doclist_dev, 'r')]
        ace_list = []
        for file_path in doclist_train_f:
            ace_info_list = extract_ace_info(file_path+ ".apf.xml")
            ace_list.extend(ace_info_list)  #追加一个列表
        return ace_list


def deal_ace_event(flag):

    # str="Former senior banker senior Callum McCarthy begins what is one of the most important jobs in London's financial world in September, when incumbent Howard Davies steps down"
    # a="senior"
    # print(str.count(a))

    event_list=encode_corpus(flag)

    for ace_event in event_list:

        for event_argument_str in ace_event.argument:
            comp_str=event_argument_str
            if ace_event.text.count(comp_str)!=1:
                print(event_argument_str+"\t\t"+str(ace_event.text.count(comp_str)))
                print(ace_event.toString())
            # print(ace_event.text.count(event_argument_str))

        # start_end_list=[]
        # for i in range(len(ace_event.argument_start)):
        #     start=ace_event.argument_start[i]
        #     end=ace_event.argument_end[i]+1
        #     # print(start,end)
        #     start_end_list.append((start,end))
        # start_end_list.sort()
        # print(start_end_list)
        # for start_end in start_end_list:
        #     for start_end_comp in start_end_list:
        #         if start_end!=start_end_comp:
        #             if start_end[1]<start_end_comp[0] or start_end[0]>start_end_comp[1]:
        #                 pass
        #             else:
        #                 print(start_end,start_end_comp)
        #                 print(ace_event.toString())

    # print("===============================start=====================================")
    # # print(event_list[1].toString())
    # ace_event=event_list[2]
    # print(ace_event.toString())
    # for i in range(len(ace_event.argument_start)):
    #     start=ace_event.argument_start[i]
    #     end=ace_event.argument_end[i]+1
    #     print(start,end)
        # print(ace_event.text[start:end])


def get_word2vec():

    wordvec = {}
    word2vec_f = open(word2vec_file, 'r')
    wordlist_f = open(wordlist_file, 'r')
    word_len = 19488
    for line in range(word_len):
        word = wordlist_f.readline().strip()
        vec = word2vec_f.readline().strip()
        temp = vec.split(',')
        temp = map(float, temp)
        vec_list = []
        for i in temp:
            vec_list.append(i)
        wordvec[word] = vec_list
    return wordvec


if __name__ == "__main__":

    deal_ace_event("test")


    # ace_list=get_ace_event_list(path)
