import os

from tf_visgen.base.data.base_data_config import BaseDataConfig

TOKEN_BEGIN = '<s>'
TOKEN_END = '</s>'
TOKEN_UNKNOWN = '<unk>'
TOKEN_PAD = '<pad>'


class MSCOCODataConfig(BaseDataConfig):
    def __init__(self, model_name="mscoco"):
        super(MSCOCODataConfig, self).__init__(model_name)
        #
        self.batch_size = 100
        self.num_threads = 8  # number of threads for data processing

        self.dim_visual_feature = 1536
        self.dim_token_feature = 512

        self.beam_size = 5
        self.num_max_bbox = 36
        self.num_visual_features = self.num_max_bbox + 1

        self.num_caption_max_length = 32

        # special token for sequence data
        self.token_start = TOKEN_BEGIN
        self.token_end = TOKEN_END
        self.token_unknown = TOKEN_UNKNOWN
        self.token_pad = TOKEN_PAD
        self.prefix_and_suffix = True

        # mscoco dataset dir
        self.mscoco_data_dir = os.path.join(self.base_data_dir, "mscoco")

        # mscoco dataset annotations dir
        self.mscoco_annotations_dir = os.path.join(
            self.mscoco_data_dir, "annotations")
        # for captions
        self.captions_train_file = os.path.join(
            self.mscoco_annotations_dir, "captions_train2014.json")
        self.captions_valid_file = os.path.join(
            self.mscoco_annotations_dir, "captions_val2014.json")
        # for instances
        self.instances_train_file = os.path.join(
            self.mscoco_annotations_dir, "instances_train2014.json")
        self.instances_valid_file = os.path.join(
            self.mscoco_annotations_dir, "instances_val2014.json")

        # for images
        self.image_dir = os.path.join(self.mscoco_data_dir, "images")
        self.train_image_dir = os.path.join(self.image_dir, "train2014")
        self.valid_image_dir = os.path.join(self.image_dir, "val2014")
        self.test_image_dir = os.path.join(self.image_dir, "test2014")

        # for embeddings
        self.embedding_dir = os.path.join(self.mscoco_data_dir, 'embeddings')
        token2vec_file_name = "token2vec_" + str(self.dim_token_feature) + ".model"
        self.token2vec_model = os.path.join(self.embedding_dir, token2vec_file_name)

        # for caption_text
        self.caption_text_dir = os.path.join(self.mscoco_data_dir, 'caption_text')
        self.caption_fixed = True
        self.caption_tokenized_txt = os.path.join(self.caption_text_dir, 'caption_tokenized.txt')
        self.vocab_file = os.path.join(self.caption_text_dir, 'vocab.txt')

        # for image region detection dir
        self.detect_dir = os.path.join(self.mscoco_data_dir, "detect")
        self.detect_train_file = os.path.join(self.detect_dir, 'detect_train.json')
        self.detect_valid_file = os.path.join(self.detect_dir, 'detect_valid.json')
        self.detect_test_file = os.path.join(self.detect_dir, 'detect_test.json')

        # for tf_records outputs
        self.tf_record_dir = os.path.join(self.mscoco_data_dir, "tf_record")
        self.train_tfrecord_dir = os.path.join(self.tf_record_dir, "train")
        self.valid_tfrecord_dir = os.path.join(self.tf_record_dir, 'valid')
        self.test_tfrecord_dir = os.path.join(self.tf_record_dir, 'test')

        # for hdf5
        self.hdf5_dir = os.path.join(self.mscoco_data_dir, 'hdf5')
        self.hdf5_train_dir = os.path.join(self.hdf5_dir, 'train')
        self.hdf5_valid_dir = os.path.join(self.hdf5_dir, 'valid')
        self.hdf5_infer_dir = os.path.join(self.hdf5_dir, 'infer')

        # for result
        self.result_dir = os.path.join(self.mscoco_data_dir, "result")
        self.train_result_dir = os.path.join(self.result_dir, "train")
        self.valid_result_dir = os.path.join(self.result_dir, 'valid')
        self.test_result_dir = os.path.join(self.result_dir, 'test')

        # for topic
        self.topic_dir = os.path.join(self.mscoco_data_dir,'topic')
        self.topic_model_file = os.path.join(self.topic_dir,'topic_lda.model')


class CaptionSplitDataConfig(MSCOCODataConfig):

    def __init__(self):
        super(CaptionSplitDataConfig, self).__init__(
            model_name="mscoco"
        )
        self.split_dir = os.path.join(self.mscoco_data_dir, 'caption_datasets')
        self.split_json = os.path.join(self.split_dir, 'dataset_coco.json')
