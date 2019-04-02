from deal_scene_graph import GenerateFeatureFile







if __name__ == '__main__':
    tfrecord_dir = ''
    read_json_dir='../test_json/test.json'
    image_dir='../test_img/'
    json_save_dir='../save_test_json/'
    generateJson = GenerateFeatureFile()
    generateJson.set_path(read_json_dir,image_dir, tfrecord_dir,json_save_dir)
    generateJson.read_scene_graph()