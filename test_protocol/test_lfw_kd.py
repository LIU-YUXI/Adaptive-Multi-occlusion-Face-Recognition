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
def accu_key(elem):
    return elem[1]

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
    # define pairs_parser_factory
    pairs_parser_factory = PairsParserFactory(pairs_file_path, args.test_set)
    # define dataloader
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False), 
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    #model def
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = iresnet100(num_features=args.embedding_size)# ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor(args.device)
    lfw_evaluator = LFWEvaluator(data_loader, pairs_parser_factory, feature_extractor)
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
        mean, std = lfw_evaluator.test(model)
        accu_list = [(os.path.basename(args.model_path), mean, std)]
    pretty_tabel = PrettyTable(["model_name", "mean accuracy", "standard error"])
    for accu_item in accu_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)
