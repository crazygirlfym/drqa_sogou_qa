#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/28
from __future__ import absolute_import
import torch
# from utils import add_argument
from corpus import WebQACorpus


def preprocess_data(args):

    # w, p, n = WebQACorpus.load_word_dictionary(args.baidu_file)
    # word_dict, pos_dict, ner_dict = WebQACorpus.load_word_dictionary(args.train_file, w, p, n)
    # word_dict.cut_by_top(args.topk)
    # torch.save([word_dict, pos_dict, ner_dict], open(args.dict_file, 'wb'))
    #
    # print(args.baidu_data)
    # print(args.train_data)
    # print(args.valid_data)
    word_dict, pos_dict, ner_dict = torch.load(args.dict_file)
    # word_dict, pos_dict, ner_dict = WebQACorpus.load_word_dictionary(args.baidu_file)

    baidu_data = WebQACorpus(args.baidu_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    train_data = WebQACorpus(args.train_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    valid_data = WebQACorpus(args.valid_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)

    print("saving baidu_data ...")
    with open(args.baidu_data, 'wb') as output:
        torch.save(baidu_data, output)
    print("saving train_data ...")
    with open(args.train_data, 'wb') as output:
        torch.save(train_data, output)
    print("saving valid_data ...")
    with open(args.valid_data, 'wb') as output:
        torch.save(valid_data, output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-baidu-file', type=str, dest="baidu_file", default="data/baidu_data.json")
    parser.add_argument('-baidu-data', type=str, dest="baidu_data", default="data/baidu_data.pt")
    parser.add_argument('-train-file', type=str, dest="train_file", default="data/sogou_shuffle_train.json")
    parser.add_argument('-train-data', type=str, dest="train_data",
                        default="data/sogou_shuffle_train.pt")
    parser.add_argument('-valid-file', type=str, dest="valid_file", default="data/sogou_shuffle_valid.json")
    parser.add_argument('-valid-data', type=str, dest="valid_data",
                        default="data/sogou_shuffle_valid.pt")
    parser.add_argument('-dict', type=str, dest="dict_file", default='data/vocab.pt')
    parser.add_argument('-topk', type=int, dest="topk", default=30000)
    # fileinput = '/home/iscas/fym/codes/sougou_train.json'
    # output = './test.json'
    args = parser.parse_args()
    # args = add_argument()
    preprocess_data(args)

    ## python preprocess.py -baidu-file=/media/iscas/linux/fym/data/pre_data/baidu_data.pt -baidu-data=/media/iscas/linux/fym/data/train_data_sogou/baidu_data.pt -train-file=/media/iscas/linux/fym/data/pre_data/train.all.pt
    # -train-data=/media/iscas/linux/fym/data/train_data_sogou/train.pt -valid-file=/media/iscas/linux/fym/data/pre_data/valid_factoid.pt -valid-data=/media/iscas/linux/fym/data/train_data_sogou/valid.pt -dict=/media/iscas/linux/fym/data/train_data_sogou/vocab.pt

