""" 
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import argparse
import yaml
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from lfw.pairs_parser import PairsParserFactory
from lfw.lfw_evaluator import LFWEvaluator
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('..')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory
from backbone.iresnet import iresnet100
import torch
import clip
import random
import numpy as np
import torch.nn as nn
def accu_key(elem):
    return elem[1]
class Adapt_Layer(torch.nn.Module):
    def __init__(self, embedding_dim, category_num):
        super(Adapt_Layer, self).__init__()
        self.embedding_dim=embedding_dim
        self.category_num=category_num
        self.linears = nn.ModuleList([nn.Linear(embedding_dim,embedding_dim) for i in range(category_num)])

    def forward(self, feature, prob):
        feature_list=[]
        for m in self.linears:
            feature_list.append(m(feature).unsqueeze(-1))
        feature_list=torch.cat(feature_list,-1)
        # print(feature_list.shape)
        pred = torch.matmul(feature_list,prob.to(dtype=feature.dtype).unsqueeze(-1)).squeeze(-1)
        return pred

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the adapt layer of face model.
        clip_model(object)
    """
    def __init__(self, backbone, adapt, clip_model, text):
        super(FaceModel, self).__init__()
        self.backbone = backbone
        self.adapt =adapt
        self.clip_model = clip_model
        self.text =text

    def forward(self, data, data_clip):
        feat = self.backbone.forward(data)
        logits_per_image, logits_per_text = self.clip_model(data_clip, self.text)
        prob = logits_per_image.softmax(dim=-1)
        # prob = torch.tensor([[1.0,0]]*data.shape[0], dtype=torch.float32).to(data.device)
        pred = self.adapt.forward(feat,prob)
        return pred

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='lfw test protocal.')
    conf.add_argument("--test_set", type = str, 
                      help = "lfw, cplfw, calfw, agedb, rfw_African, \
                      rfw_Asian, rfw_Caucasian, rfw_Indian.")
    conf.add_argument("--data_conf_file", type = str, 
                      help = "the path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Resnet, Mobilefacenets..")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type = int, default = 1024)
    conf.add_argument('--model_path', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of model or the directory which some models in.')
    conf.add_argument('--adapt_path', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of adapt layer or the directory which some models in.')
    conf.add_argument('--embedding_size', type = int, default = 512, 
                      help='embedding size of backbones.')
    conf.add_argument("--device", type = str, default='cuda:0',
                      help = "dviece.")
    args = conf.parse_args()
    # parse config.
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)[args.test_set]
        pairs_file_path = data_conf['pairs_file_path']
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file_path = data_conf['image_list_file_path']

    clip_model, preprocess = clip.load("RN50x16", device=torch.device(args.device))
    clip_model.eval()
    text = clip.tokenize(["A human face","A human face in a mask" ]).to(args.device)
    args.category_mum = text.shape[0]
    print("mask category num:",args.category_mum)

    # define pairs_parser_factory
    pairs_parser_factory = PairsParserFactory(pairs_file_path, args.test_set)
    # define dataloader
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False, preprocess), 
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    #model def
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = iresnet100(num_features=args.embedding_size)# ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor(args.device)
    lfw_evaluator = LFWEvaluator(data_loader, pairs_parser_factory, feature_extractor)

    adapt_model = Adapt_Layer(args.embedding_size,args.category_mum)
    if os.path.isdir(args.model_path):
        accu_list = []
        model_name_list = os.listdir(args.model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                model_path = os.path.join(args.model_path, model_name)
                model_loader.load_state_dict(torch.load(model_path))
                model = model_loader.cuda(args.device)
                mean, std = lfw_evaluator.test(model)
                accu_list.append((os.path.basename(model_path), mean, std))
        accu_list.sort(key = accu_key, reverse=True)
    else:
        model_loader.load_state_dict(torch.load(args.model_path,map_location='cpu')['state_dict'])
        model = model_loader.cuda(args.device)
        adapt_model.load_state_dict(torch.load(args.adapt_path,map_location='cpu')['state_dict'])
        adapt_model = adapt_model.cuda(args.device)
        test_model = FaceModel(model,adapt_model,clip_model,text)
        mean, std = lfw_evaluator.test(test_model)
        accu_list = [(os.path.basename(args.model_path), mean, std)]
    pretty_tabel = PrettyTable(["model_name", "mean accuracy", "standard error"])
    for accu_item in accu_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)
