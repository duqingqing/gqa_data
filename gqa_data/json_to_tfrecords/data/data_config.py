# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

home = str(Path.home())

BATCH_SIZE = 100
BASE_DATA_DIR = os.path.join(home, '/media/zutnlp/49741bd8-e3dd-4326-bf71-0394aa198e95/zutnlpcv/gqa_tfrecords/')  # base data dir"/home/liuxiaoming/data/visualgenome"
PHRASE_MAX_LENGTH = 64

TOKEN_BEGIN = '<s>'
TOKEN_END = '</s>'
TOKEN_UNKNOWN = '<unk>'
TOKEN_PAD = '<pad>'


def get_vocab_file(dir,dim):
    return
class VisualGenomeDataConfig(object):
    """
    Visual Genome Data sets Config Class
    """

    batch_size = BATCH_SIZE
    base_data_dir = BASE_DATA_DIR
    phrase_max_length = PHRASE_MAX_LENGTH

    image_dir = os.path.join(base_data_dir, "images")

    # json data file provided by visual genome dataset
    image_data_json = os.path.join(base_data_dir, "image_data.json")
    objects_json = os.path.join(base_data_dir, "objects.json")
    attributes_json = os.path.join(base_data_dir, "attributes.json")
    relationships_json = os.path.join(base_data_dir, "relationships.json")
    region_descriptions_json = os.path.join(base_data_dir, "region_descriptions.json")
    region_graphs_json = os.path.join(base_data_dir, "region_graphs.json")

    #json scene_graph data make dy dqq
    scene_graphs_json = os.path.join(base_data_dir,"feature_json/")


    # for statistics
    statistics_dir = os.path.join(base_data_dir, "statistics")

    # for embeddings
    embeddings_dir = os.path.join(base_data_dir, "embeddings")
    region_phrase_txt = os.path.join(embeddings_dir, "region_phrase.txt")

    vocab_file = os.path.join(embeddings_dir, "vocab.txt")

    # for special token
    token_start = TOKEN_BEGIN
    token_end = TOKEN_END
    token_unknown = TOKEN_UNKNOWN
    token_pad = TOKEN_PAD

    # for feature
    feature_dir = os.path.join(base_data_dir, "features")
    image_feature_file = os.path.join(feature_dir, "image_feature.mat")
    image_feature_tf_file = os.path.join(feature_dir, "image_feature.tfrecord")

    #for train data
    train_data_dir = os.path.join(base_data_dir)


