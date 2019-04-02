# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import ijson.backends.yajl2_cffi as ijson
import tensorflow as tf

from visgen.data.data_config import VisualGenomeDataConfig
from visgen.data.data_utils import VisualGenomeDataUtils
from visgen.models import Image, Attribute, Relationship, Object

TOKEN_BEGIN = '<S>'
TOKEN_END = '</S>'


class VisualGenomeDataLoader(object):
    """
    Visual Genome Raw Dataset Loader

    """

    def __init__(self):
        self.config = VisualGenomeDataConfig()
        self.data_utils = VisualGenomeDataUtils()

    # load image meta data
    def load_images(self):
        """
        load image json data
        :return:
            yield data generator : (id_batch,shape_batch, raw_data_batch)
        """

        data_batch = []
        with open(file=self.config.image_data_json, mode='rb') as f:
            data_gen = ijson.items(f, "item")
            for data in data_gen:
                image_data = Image(image_id=data['image_id'], url=data['url'],
                                   width=data['width'], height=data['height'],
                                   coco_id=data['coco_id'], flickr_id=data['flickr_id'])
                # image_rawdata = self.data_utils.load_image_raw(image_data.image_id)
                data_batch.append(image_data)

                if len(data_batch) == self.config.batch_size:
                    yield data_batch
                    data_batch = []

        if len(data_batch) > 0:
            yield data_batch
        del data_batch

    def load_regions(self):
        batch_data = []
        with open(file=self.config.region_descriptions_json, mode='rb') as f:
            data_gen = ijson.items(f, "item")
            for data in data_gen:
                image_id = data['id']
                regions = data['regions']
                for r in regions:
                    batch_data.append(r)
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data
            del batch_data

    def load_image_regions(self):
        """
        load image with regions
        :return:
        """
        batch_data = []
        with open(file=self.config.region_descriptions_json, mode='rb') as f:
            items = ijson.items(f, "item")
            for data in items:  # data is the json of image with regions
                batch_data.append(data)
                if len(batch_data) == self.config.batch_size:
                    yield batch_data
                    batch_data = []

        if len(batch_data) > 0:
            yield batch_data
        del batch_data

    # load objects in a given image by image_id
    def load_image_objects(self, image_id):
        result = []
        with open(file=self.config.objects_json, mode='rb') as f:
            item_gen = ijson.items(f, "item")
            for item in item_gen:
                if image_id == item['image_id']:
                    objects = item["objects"]
                    for obj_json in objects:
                        object_id = obj_json["object_id"]
                        h = obj_json["h"]
                        w = obj_json["w"]
                        x = obj_json["x"]
                        y = obj_json["y"]
                        names = obj_json["names"]
                        synsets = obj_json["synsets"]
                        obj = Object(image_id=image_id, object_id=object_id,
                                     x=x, y=y, height=h, width=w, names=names, synsets=synsets)
                        result.append(obj)
                    break
        return result

    def load_objects(self):
        batch_data = []
        with open(file=self.config.objects_json, mode='rb') as f:
            item_gen = ijson.items(f, "item")
            for item in item_gen:
                image_id = item["image_id"]
                objects = item["objects"]
                for object_dict in objects:
                    object_dict['image_id'] = image_id
                    batch_data.append(object_dict)
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data

            del batch_data

    def load_attribute(self):
        batch_data = []
        with open(file=self.config.attributes_json, mode='rb') as f:
            item_generator = ijson.items(f, "item")
            for data in item_generator:
                image_id = data["image_id"]
                attributes = data["attributes"]
                for obj_json in attributes:
                    object_id = obj_json["object_id"]
                    subject = obj_json["subject"]
                    attribute = obj_json["attribute"]
                    synset = obj_json["synset"]
                    att = Attribute(object_id=object_id, subject=subject, attribute=attribute, synset=synset)
                    batch_data.append(att)
                    print("attribute={}".format(attribute))
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data
            del batch_data

    def load_relationship(self):
        batch_data = []
        with open(file=self.config.relationships_json, mode='rb') as f:
            item_generator = ijson.items(f, "item")
            for data in item_generator:
                image_id = data["image_id"]
                relationship_jsons = data["relationships"]
                for r_json in relationship_jsons:
                    relationship_id = r_json["relationship_id"]
                    predicate = r_json["predicate"]
                    synsets = r_json["synsets"]
                    subject = Object.from_json(image_id=image_id, object_json=r_json["subject"])
                    object = Object.from_json(image_id=image_id, object_json=r_json["object"])
                    relationship = Relationship(image_id=image_id, relationship_id=relationship_id,
                                                subject=subject, object=object,
                                                predicate=predicate, synsets=synsets)
                    batch_data.append(relationship)
                    # print("image_id={}, relationship={}".format(image_id, relationship))
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []

            if len(batch_data) > 0:
                yield batch_data
            del batch_data

    def load_region_graphs(self):
        """
        load region graph data
        :return:
        """
        batch_data = []
        with open(file=self.config.region_graphs_json, mode='rb') as f:
            item_gen = ijson.items(f, "item")
            for item in item_gen:
                image_id = item["image_id"]
                regions = item["regions"]
                batch_data.append((image_id, regions))
                if len(batch_data) == self.config.batch_size:
                    yield batch_data
                    batch_data = []
        if len(batch_data) > 0:
            yield batch_data
        del batch_data

    def load_object_class_count(self, min_class_num, max_class_num):
        """
            load class count in objects {$class_name:$class_count}
                between min_class_num and max_class_num object instances
        :param min_class_num:
        :param max_class_num:
        :return:
        """

        class_count_file = os.path.join(self.config.statistics_dir, 'classes_instance_count.txt')
        class_count_dict = dict()
        with open(file=class_count_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                splits = str.split(l, '\t')
                class_name = splits[0]
                class_num = int(splits[1])
                if min_class_num <= class_num <= max_class_num:
                    class_count_dict[class_name] = class_num
        return class_count_dict


def main(_):
    print("main function")
    visgen_loader = VisualGenomeDataLoader()

    # for data loader
    data_gen = visgen_loader.load_relationship()
    for batch, batch_data in enumerate(data_gen):
        for data in batch_data:
            print("data={}".format(data))

    # for embedding building
    visgen_loader.build_phrase_txt()
    visgen_loader.build_embeddings()

    pass


if __name__ == '__main__':
    tf.app.run()
