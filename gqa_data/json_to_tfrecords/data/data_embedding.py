# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import re
from gensim import corpora
import nltk
import numpy as np
from gensim.models.word2vec import Word2Vec

from visgen.data.data_config import VisualGenomeDataConfig
from visgen.data.data_loader import VisualGenomeDataLoader
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_config = VisualGenomeDataConfig()
data_loader = VisualGenomeDataLoader()

WITH_BEGIN_END_TOKEN = True


class VisualGenomeDataEmbedding(object):
    """
    embedding for visual genome data
    """

    def __init__(self, with_begin_end_token=WITH_BEGIN_END_TOKEN):
        self.with_begin_end_token = with_begin_end_token
        self.data_config = data_config

    def load_embeddings(self, embedding_dim):
        """
        loading embeddings for words in all phrases
        :param embedding_dim: {50,100,200,300}
        :return:
        """
        w2v_model_file = os.path.join(data_config.embeddings_dir,
                                      'phrase_word2vec_' + str(embedding_dim) + '.model')
        if not os.path.isfile(w2v_model_file):
            self.build_embeddings()

        token_unknown = self.data_config.token_unknown

        w2v_model = Word2Vec.load(w2v_model_file)
        vocab_size = len(w2v_model.wv.index2word)  # initial vocab_size

        # +1 for unknown token
        token_embedding_matrix = np.zeros([vocab_size + 1, embedding_dim])

        word2index = dict()
        index2word = dict()
        for idx, word in enumerate(w2v_model.wv.index2word):
            word_embedding = w2v_model.wv[word]
            index2word[idx] = word
            word2index[word] = idx
            token_embedding_matrix[idx] = word_embedding

        # for unknown word
        token_embedding_matrix[vocab_size] = np.zeros([embedding_dim])
        word2index[token_unknown] = idx + 1
        index2word[idx + 1] = token_unknown

        return token_embedding_matrix, word2index, index2word

    def _process_caption(self, caption):
        """Processes a caption string into a list of tonenized words.

        Args:
          caption: A string caption.

        Returns:
          A list of strings; the tokenized caption.
        """
        caption = str.strip(caption)
        if self.with_begin_end_token:  # phrase sentence with <s> and </s> token
            tokenized_caption = [self.data_config.token_start]
            tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
            tokenized_caption.append(self.data_config.token_end)
        else:
            tokenized_caption = nltk.tokenize.word_tokenize(caption.lower())
        return tokenized_caption

    def build_phrase_txt(self):
        """
        build all phrase sentences into txt file, and this txt could be used as sources of embeddings

        :return:
        """
        region_gen = data_loader.load_regions()
        begin_token = data_config.token_start
        end_token = data_config.token_end
        phrase_list = []
        count = 0
        region_phrase_txt = data_config.region_phrase_txt
        with open(file=region_phrase_txt, mode='a', encoding='utf-8') as f:
            for batch_num, batch_data in enumerate(region_gen):
                for data in batch_data:
                    phrase = data['phrase']
                    phrase_line = self._process_caption(phrase)
                    phrase_line = " ".join(phrase_line)+"\n"
                    phrase_list.append(phrase_line)
                    if len(phrase_list) == 10000:
                        f.writelines(phrase_list)
                        count += len(phrase_list)
                        print('append {} phrases to file {}'.format(count, region_phrase_txt))
                        phrase_list = []
            if len(phrase_list) > 0:
                f.writelines(phrase_list)
                count += len(phrase_list)
                print('append {} phrases to file {}'.format(count, region_phrase_txt))

    def build_embeddings(self):
        """
        build embeddings based on phrase sentence txt file from regions
        :return:
        """
        if not os.path.isfile(data_config.region_phrase_txt):
            self.build_phrase_txt()

        with open(data_config.region_phrase_txt, "r") as f:
            lines = f.readlines()
            sentences = [str.split(line) for line in lines]
            for size in [50, 100, 200, 300]:
                model_file_path = os.path.join(data_config.embeddings_dir, "phrase_word2vec_" + str(size) + ".model")
                print("begin building {}".format(model_file_path))
                model = Word2Vec(sentences, size=size, window=5, min_count=1, workers=4)
                model.save(model_file_path)

                print("end building {}".format(model_file_path))

        pass

    def build_vocab(self):
        filename = self.data_config.region_phrase_txt
        with open(file=filename, mode='r', encoding='utf-8') as f:
            sentences = f.readlines()
            texts = [[token for token in sentence.split()]
                     for sentence in sentences]
        frequency = nltk.defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [[token for token in text if frequency[token] >= 4]
                 for text in texts]
        dictionary = corpora.Dictionary(texts)
        vocab_file = self.data_config.vocab_file
        dictionary.save_as_text(fname=vocab_file)

if __name__ == '__main__':
    data_embedding = VisualGenomeDataEmbedding()
    # data_embedding.build_phrase_txt()
    # data_embedding.build_embeddings()
    data_embedding.build_vocab()
