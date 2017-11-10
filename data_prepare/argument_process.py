from data_prepare.xml_parse import xml_parse_base
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
sourcepath = "D:/Code/pycharm/Sequence-Label-Attention/data/"
acepath=sourcepath+"/ace_en_source/"

doclist_train = sourcepath + "split1.0/ACE_train_filelist"
doclist_test = sourcepath + "split1.0/ACE_test_filelist"
doclist_dev = sourcepath + "split1.0/ACE_dev_filelist"
doclist_full = sourcepath + "split1.0/ACE_full_filelist"

data_save_path = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/argument_data.data"
form_data_save_path = "D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/argument_data_form.data"

argument_save_txt="D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/argument_raw/argument_"

word2vec_file = sourcepath+'/word2vec/wordvector'
wordlist_file = sourcepath+'/word2vec/wordlist'

ACE_EVENT_Trigger_Type=[None, 'Trigger_Transport', 'Trigger_Elect', 'Trigger_Start-Position', 'Trigger_Nominate', 'Trigger_Attack', 'Trigger_End-Position', 'Trigger_Meet', 'Trigger_Marry', 'Trigger_Phone-Write', 'Trigger_Transfer-Money', 'Trigger_Sue', 'Trigger_Demonstrate', 'Trigger_End-Org', 'Trigger_Injure', 'Trigger_Die', 'Trigger_Arrest-Jail', 'Trigger_Transfer-Ownership', 'Trigger_Start-Org', 'Trigger_Execute', 'Trigger_Trial-Hearing', 'Trigger_Sentence', 'Trigger_Be-Born', 'Trigger_Charge-Indict', 'Trigger_Convict', 'Trigger_Declare-Bankruptcy', 'Trigger_Release-Parole', 'Trigger_Fine', 'Trigger_Pardon', 'Trigger_Appeal', 'Trigger_Merge-Org', 'Trigger_Extradite', 'Trigger_Divorce', 'Trigger_Acquit']
ACE_EVENT_Argument_Type=[None, 'Argument_Vehicle', 'Argument_Artifact', 'Argument_Destination', 'Argument_Agent', 'Argument_Person', 'Argument_Position', 'Argument_Entity', 'Argument_Attacker', 'Argument_Place', 'Argument_Time-At-Beginning', 'Argument_Target', 'Argument_Giver', 'Argument_Recipient', 'Argument_Plaintiff', 'Argument_Money', 'Argument_Victim', 'Argument_Time-Within', 'Argument_Buyer', 'Argument_Time-Ending', 'Argument_Instrument', 'Argument_Seller', 'Argument_Origin', 'Argument_Time-Holds', 'Argument_Org', 'Argument_Time-At-End', 'Argument_Time-Before', 'Argument_Time-Starting', 'Argument_Time-After', 'Argument_Beneficiary', 'Argument_Defendant', 'Argument_Adjudicator', 'Argument_Sentence', 'Argument_Crime', 'Argument_Prosecutor', 'Argument_Price']


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

    doc = minidom.parse(apf_file)
    root = doc.documentElement

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
            #text开始均为0
            for charse in mention_ldc_scope_charseq:
                start = xml_parse_base.get_attrvalue(None,charse, 'START')#charseq的START属性值
                end = xml_parse_base.get_attrvalue(None, charse, 'END')
                s = start
                R_element.text_start = 0
                R_element.text_end = int(end)-int(start)


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
            for mention_argument in mention_arguments:
                mention_argument_extent = xml_parse_base.get_xmlnode(None, mention_argument, 'extent')
                mention_argument_charseq = xml_parse_base.get_xmlnode(None, mention_argument_extent[0], 'charseq')
                for argument_charse in mention_argument_charseq:
                    start = xml_parse_base.get_attrvalue(None,argument_charse, 'START')
                    end = xml_parse_base.get_attrvalue(None, argument_charse, 'END')

                R_element.argument_start.append(int(start) - int(s))#多个事件元素
                R_element.argument_end.append( int(end) - int(s))
                R_element.argument.append(R_element.text[R_element.argument_start[i]:R_element.argument_end[i]+1])
                R_element.argument_type.append(xml_parse_base.get_attrvalue(None, mention_argument, 'ROLE'))#事件元素类型
                i+=1

            R.append(R_element)

    return R


    # # 存储事件实体的list
    # R = []
    # doc = minidom.parse(apf_file)#从xml文件得到doc对象
    # root = doc.documentElement#获得根对象source_file
    # entity = {}
    # # 获取实体提及
    # entity_nodes = xml_parse_base.get_xmlnode(None, root, 'entity')
    # for entity_node in entity_nodes:
    #     entity_type = xml_parse_base.get_attrvalue(None, entity_node, 'SUBTYPE')
    #     entity_mention_nodes = xml_parse_base.get_xmlnode(None, entity_node, 'entity_mention')
    #     for entity_mention_node in entity_mention_nodes:
    #         entity_mention_id = xml_parse_base.get_attrvalue(None, entity_mention_node, 'ID')
    #         entity_mention_head = xml_parse_base.get_xmlnode(None, entity_mention_node, 'head')
    #         entity_mention_head_charseq = xml_parse_base.get_xmlnode(None, entity_mention_head[0], 'charseq')
    #         for charse in entity_mention_head_charseq:
    #             entity_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
    #             entity_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
    #             entity[entity_mention_id] = [entity_mention_start, entity_mention_end, entity_type]
    # #获得value提及
    # value_nodes = xml_parse_base.get_xmlnode(None, root, 'value')
    # for value_node in value_nodes:
    #     value_type = xml_parse_base.get_attrvalue(None, value_node, 'TYPE')
    #     value_mention_nodes = xml_parse_base.get_xmlnode(None, value_node, 'value_mention')
    #     for value_mention_node in value_mention_nodes:
    #         value_mention_id = xml_parse_base.get_attrvalue(None, value_mention_node, 'ID')
    #         value_mention_extent = xml_parse_base.get_xmlnode(None, value_mention_node, 'extent')
    #         value_mention_extent_charseq = xml_parse_base.get_xmlnode(None, value_mention_extent[0], 'charseq')
    #         for charse in value_mention_extent_charseq:
    #             value_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
    #             value_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
    #             entity[value_mention_id] = [value_mention_start,value_mention_end,value_type]
    # #获得time提及
    # timex2_nodes = xml_parse_base.get_xmlnode(None, root, 'timex2')
    # for timex2_node in timex2_nodes:
    #     timex2_mention_nodes = xml_parse_base.get_xmlnode(None, timex2_node, 'timex2_mention')
    #     for timex2_mention_node in timex2_mention_nodes:
    #         timex2_mention_id = xml_parse_base.get_attrvalue(None, timex2_mention_node, 'ID')
    #         timex2_mention_extent = xml_parse_base.get_xmlnode(None, timex2_mention_node, 'extent')
    #         timex2_mention_extent_charseq = xml_parse_base.get_xmlnode(None, timex2_mention_extent[0], 'charseq')
    #         for charse in timex2_mention_extent_charseq:
    #             timex2_mention_start = xml_parse_base.get_attrvalue(None,charse, 'START')
    #             timex2_mention_end = xml_parse_base.get_attrvalue(None, charse, 'END')
    #             entity[timex2_mention_id] = [timex2_mention_start,timex2_mention_end,'timex2']
    #
    #
    # event_nodes = xml_parse_base.get_xmlnode(None, root, 'event')#获得对象名为event的节点列表
    # for node in event_nodes:
    #     # 获取事件mention
    #     mention_nodes = xml_parse_base.get_xmlnode(None, node, 'event_mention')#获得对象名为event_mention的节点列表
    #     for mention_node in mention_nodes:
    #         R_element = ACE_info()
    #         # 获取事件id
    #         R_element.id = xml_parse_base.get_attrvalue(None, mention_node, 'ID')#获得event_mention的ID属性值
    #         # 获取事件子类型
    #         R_element.trigger_sub_type = xml_parse_base.get_attrvalue(None, node, 'SUBTYPE')#获得event的SUBTYPE属性值
    #         # 获取事件所在语句
    #         mention_ldc_scope = xml_parse_base.get_xmlnode(None, mention_node, 'ldc_scope')#获得ldc_scope列表
    #         mention_ldc_scope_charseq = xml_parse_base.get_xmlnode(None, mention_ldc_scope[0], 'charseq')
    #         #获得事件语句并将/n换位空格
    #         text_str = xml_parse_base.get_nodevalue(None, mention_ldc_scope_charseq[0], 0).replace("\n", " ")
    #         R_element.text = text_str
    #         s = None
    #         m = 0
    #         #text开始均为0
    #         for charse in mention_ldc_scope_charseq:
    #             start = xml_parse_base.get_attrvalue(None,charse, 'START')#charseq的START属性值
    #             end = xml_parse_base.get_attrvalue(None, charse, 'END')
    #             s = start
    #             R_element.text_start = 0
    #             R_element.text_end = int(end)-int(start)
    #             for j, x in enumerate(entity):
    #                 #print entity[x][0]
    #                 if int(entity[x][0])>=int(start) and int(entity[x][1])<=int(end):
    #                     R_element.entity_start.append(int(entity[x][0]) - int(s))
    #                     #print R_element.entity_start
    #                     R_element.entity_end.append(int(entity[x][1]) - int(s))
    #                     R_element.entity_type.append(entity[x][2])
    #                     R_element.entity.append(R_element.text[R_element.entity_start[m]:R_element.entity_end[m]+1])
    #                     m+=1
    #
    #
    #         # 获取事件触发词
    #         mention_anchor = xml_parse_base.get_xmlnode(None, mention_node, 'anchor')#获得anchor列表
    #         mention_anchor_charseq = xml_parse_base.get_xmlnode(None, mention_anchor[0], 'charseq')
    #         for anch in mention_anchor_charseq:
    #             start = xml_parse_base.get_attrvalue(None, anch, 'START')
    #             end = xml_parse_base.get_attrvalue(None, anch, 'END')
    #             R_element.trigger_start = int(start)-int(s)#相对语句位置
    #             R_element.trigger_end = int(end)-int(s)
    #             R_element.trigger = R_element.text[R_element.trigger_start:R_element.trigger_end+1]#begin 这样的
    #
    #
    #
    #         # 获取事件元素
    #         mention_arguments = xml_parse_base.get_xmlnode(None, mention_node, 'event_mention_argument')#event_mention_argument列表
    #         i = 0
    #         arg = []
    #         for mention_argument in mention_arguments:
    #             mention_argument_refid = xml_parse_base.get_attrvalue(None,mention_argument, 'REFID')
    #             #mention_argument_extent = get_xmlnode(None, mention_argument, 'extent')
    #             #mention_argument_charseq = get_xmlnode(None, mention_argument_extent[0], 'charseq')
    #             try:
    #                 argument_position = entity[mention_argument_refid]
    #                 start = argument_position[0]
    #                 end = argument_position[1]
    #                 entity_type = argument_position[2]
    #             except KeyError:
    #                 print ('error')
    #                 mention_argument_extent = xml_parse_base.get_xmlnode(None, mention_argument, 'extent')
    #                 mention_argument_charseq = xml_parse_base.get_xmlnode(None, mention_argument_extent[0], 'charseq')
    #                 for argument_charse in mention_argument_charseq:
    #                     start = xml_parse_base.get_attrvalue(None,argument_charse, 'START')
    #                     end = xml_parse_base.get_attrvalue(None, argument_charse, 'END')
    #                     entity_type = None
    #
    #             R_element.argument_start.append(int(start) - int(s))#多个事件元素
    #             R_element.argument_end.append( int(end) - int(s))
    #             R_element.argument.append(R_element.text[R_element.argument_start[i]:R_element.argument_end[i]+1])
    #             #arg.append(get_nodevalue(None, mention_argument_charseq[0], 0).replace("\n", " "))#事件元素列表
    #             R_element.argument_entity_type.append(entity_type)
    #             R_element.argument_type.append(xml_parse_base.get_attrvalue(None, mention_argument, 'ROLE'))#事件元素类型
    #             i+=1
    #
    #         R.append(R_element)
    # return R


def encode_corpus(flag):
    if flag=="train":
        doclist_f=[acepath + i.replace('\n', '') for i in open(doclist_train, 'r')]
        ace_list = []
        for file_path in doclist_f:
            ace_info_list = extract_ace_info(file_path+ ".apf.xml")
            ace_list.extend(ace_info_list)  #追加一个列表
        return ace_list
    if flag == "test":
        doclist_f=[acepath + i.replace('\n', '') for i in open(doclist_test, 'r')]
        ace_list = []
        for file_path in doclist_f:
            ace_info_list = extract_ace_info(file_path+ ".apf.xml")
            ace_list.extend(ace_info_list)  #追加一个列表
        return ace_list
    if flag == "dev":
        doclist_f=[acepath + i.replace('\n', '') for i in open(doclist_dev, 'r')]
        ace_list = []
        for file_path in doclist_f:
            ace_info_list = extract_ace_info(file_path+ ".apf.xml")
            ace_list.extend(ace_info_list)  #追加一个列表
        return ace_list
    # 如果都不是返回全部
    doclist_f=[acepath + i.replace('\n', '') for i in open(doclist_full, 'r')]
    ace_list = []
    for file_path in doclist_f:
        ace_info_list = extract_ace_info(file_path+ ".apf.xml")
        ace_list.extend(ace_info_list)  #追加一个列表
    return ace_list


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


def argu_split_sentence(sentence_list,tag_list,argument_type,num_start,num_end):
    assert len(sentence_list)==len(tag_list),"len(sentence_list)!=len(tag_list)"
    assert num_start<num_end,"num_start<num_end"
    split_list = []
    tag_split_list=[]
    if type(sentence_list) == list:
        split_index=-1
        split_start=0
        split_end=0
        for index,sen in enumerate(sentence_list):
            split_start=split_end
            split_end+=len(sen)
            if num_start>=split_start and num_end<=split_end:
                split_index=index
                break
        split_list.extend(sentence_list[:split_index])
        tag_split_list.extend(tag_list[:split_index])

        num_start-=split_start
        num_end-=split_start
        sentence=sentence_list[split_index]

        split_list.append(sentence[:num_start])
        tag_split_list.append(None)
        split_list.append(sentence[num_start:num_end])
        tag_split_list.append(argument_type)
        split_list.append(sentence[num_end:])
        tag_split_list.append(None)

        split_list.extend(sentence_list[split_index+1:])
        tag_split_list.extend(tag_list[split_index+1:])

    return split_list,tag_split_list


"""
处理ACE事件
"""
def deal_ace_event(flag):

    event_list=encode_corpus(flag)
    print(len(event_list))
    # event_file = open(argument_save_txt+flag, 'w')

    word_result_list=[]
    tag_result_list=[]
    for ace_event in event_list:

        record_flag=True

        sen_split_list=[]
        tag_split_list=[]
        sen_split_list.append(ace_event.text)
        tag_split_list.append(None)

        for i,event_argument in enumerate(ace_event.argument):
            sen_split_list,tag_split_list=argu_split_sentence(sen_split_list,tag_split_list,"Argument_"+ace_event.argument_type[i],ace_event.argument_start[i],ace_event.argument_end[i]+1)

        sen_split_list, tag_split_list = argu_split_sentence(sen_split_list, tag_split_list, "Trigger_"+ace_event.trigger_sub_type,
                                                             ace_event.trigger_start, ace_event.trigger_end + 1)

        if ''.join(sen_split_list)!=ace_event.text:
            record_flag=False
            print(sen_split_list)
        else:
            for event_argument in ace_event.argument:
                if event_argument not in sen_split_list:
                    record_flag=False

        if record_flag:
            word_result, tag_result = get_word_tag_list(sen_split_list, tag_split_list)
            word_result_list.append(word_result)
            tag_result_list.append(tag_result)

    print(len(word_result_list))
    print(len(tag_result_list))
    return word_result_list,tag_result_list



        # if record_flag:
        #     event_file.write(str(sen_split_list))
        #     event_file.write('\n')
        #     event_file.write(str(tag_split_list))
        #     event_file.write('\n')
        # else:
        #     print(ace_event.toString())

    # event_file.close()


# def read_event_file(flag):
#     event_file = open(argument_save_txt + flag, 'r')
#     for index,line in enumerate(event_file.readlines()):
#         sen_split_list=line.strip()
#         print(sen_split_list)
#         print(type(sen_split_list))
        # for i in sen_split_list:
        #     print(i)
    # for index,line in enumerate(event_file):
    #     sen_split_list=line
    #     print(sen_split_list)
    #     if type(sen_split_list) == list:
    #         print("111111111111")
        # for i in sen_split_list:
        #     print(i)
        # print(index,sen_split_list)

def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`<>]-", " ", string)
    string = re.sub(r"\'m", r" 'm", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\,", r" , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    # string=number_form(string)
    return string.strip() if TREC else string.strip().lower()


def get_word_tag_list(sen_list,tag_list):
    word_result_list=[]
    tag_result_list=[]

    for index,(phrase,tag) in enumerate(zip(sen_list,tag_list)):
        phrase=clean_str(phrase).split()
        for word in phrase:
            word_result_list.append(word)
            tag_result_list.append(tag)

    return word_result_list,tag_result_list


# def get_dot_word():
#
#     wordlist_f=open(wordlist_file,'r')
#     word_dot_list=dict()
#     for line in wordlist_f:
#         word=line.strip()
#         if "." in word:
#             if "."!=word and "..."!=word:
#                 temp=word
#                 word_dot_list[temp.replace("."," <dot> ")]=word
#     return word_dot_list
#
# word_dot_list=get_dot_word()
#
# def number_form(s):
#     num_list = re.findall("\d+\s,\s\d+", s)
#     for re_num in num_list:
#         s = s.replace(re_num, re_num.replace(" ", ""))
#
#     print("=================================")
#     print(s)
#     if s in word_dot_list.keys():
#         print(s)
#         print("1111111111111111111111111")
#         s=word_dot_list.get(s)
#     return s


def list_to_vec(word_result_list, tag_result_list):
    for word_list,tag_list in zip(word_result_list, tag_result_list):
        print(word_list,tag_list)



if __name__ == "__main__":
    # sen_list=['Former senior banker ', 'Callum McCarthy', ' ', 'begins', ' what is ',"one of the most important jobs in London's financial world", ' in ', 'September', ', when incumbent Howard Davies steps down']
    # tag_list=[None, 'Argument_Person', None, 'Trigger_Start-Position', None, 'Argument_Position', None, 'Argument_Time-Within',None]

    # sen_list =['', 'Iraqis', ' mostly ', 'fought', ' back with small ', 'arms', ', ', 'pistols', ', machine ', 'guns',' and rocket-propelled ', 'grenades', '']
    # tag_list =[None, 'Argument_Attacker', None, 'Trigger_Attack', None, 'Argument_Instrument', None, 'Argument_Instrument', None, 'Argument_Instrument', None, 'Argument_Instrument', None]
    #
    # sen_list=['The ', 'EU', ' is set to ', 'release', ' ', '20 million euros (US$21.5 million',') in immediate humanitarian aid for ', 'Iraq'," if war breaks out and may dip into an ``emergency reserve'' of 250 million euros (US$269 million) for humanitarian relief"]
    # tag_list=[None, 'Argument_Giver', None, 'Trigger_Transfer-Money', None, 'Argument_Money', None, 'Argument_Beneficiary', None]

    # word_result_list, tag_result_list=get_word_tag_list(sen_list,tag_list)

    # print(sen_list)
    # print(tag_list)
    #
    # word_result_list=[]
    # tag_result_list=[]
    #
    # for index,(phrase,tag) in enumerate(zip(sen_list,tag_list)):
    #     phrase=clean_str(phrase).split()
    #     for word in phrase:
    #         word_result_list.append(word)
    #         tag_result_list.append(tag)
    #     # print(index,phrase,tag)
    # print(word_result_list)
    # print(tag_result_list)


    # pass

    # read_event_file("test")

    # deal_ace_event("dev")
    # deal_ace_event("train")
    # word_result_list, tag_result_list=deal_ace_event("test")
    # list_to_vec(word_result_list, tag_result_list)
    # print(word_result_list)
    # print(tag_result_list)

    ace_info_list = extract_ace_info("D:\Code\pycharm\Sequence-Label-Attention\data\AFP_ENG_20030401.0476.apf.xml")
    for ace_event in ace_info_list:
        if "best-known" in ace_event.text:
            print("=============================================")
            print(ace_event.toString())
            record_flag=True

            sen_split_list=[]
            tag_split_list=[]
            sen_split_list.append(ace_event.text)
            tag_split_list.append(None)
            for i,event_argument in enumerate(ace_event.argument):
                sen_split_list,tag_split_list=argu_split_sentence(sen_split_list,tag_split_list,"Argument_"+ace_event.argument_type[i],ace_event.argument_start[i],ace_event.argument_end[i]+1)

            sen_split_list, tag_split_list = argu_split_sentence(sen_split_list, tag_split_list, "Trigger_"+ace_event.trigger_sub_type,
                                                                 ace_event.trigger_start, ace_event.trigger_end + 1)

            if ''.join(sen_split_list)!=ace_event.text:
                record_flag=False
                print(sen_split_list)
                print(''.join(sen_split_list))
                print(ace_event.text)
            else:
                print(ace_event.text)


    # ace_info_list = extract_ace_info(acepath + "AFP_ENG_20030401.0476.apf.xml")
    # for ace_event in ace_info_list:
    #     sen_split_list=[]
    #     tag_split_list=[]
    #     sen_split_list.append(ace_event.text)
    #     tag_split_list.append(None)
    #     for i,event_argument in enumerate(ace_event.argument):
    #         sen_split_list,tag_split_list=argu_split_sentence(sen_split_list,tag_split_list,"Argument_"+ace_event.argument_type[i],ace_event.argument_start[i],ace_event.argument_end[i]+1)
    #
    #     print(ace_event.toString())
    #     print(sen_split_list)
    #     print(tag_split_list)
    #     print("============================================================")
    #     sys.exit()
    #     print("=========================================================")
    #     print(''.join(sen_split_list))
    #     if ''.join(sen_split_list)!=ace_event.text:
    #         if ''.join(sen_split_list)==ace_event.text+ace_event.text:
    #             print("11111111")
    #         else:
    #             print("2222222")
    #
    #         print(ace_event.toString())
    #         print(''.join(sen_split_list))
    #         print(ace_event.text)
    #         print("====================================================")
    #
    #
    #
    # ace_list=get_ace_event_list(path)
