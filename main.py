
import os,json,shutil,io,hashlib,subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
from pycocotools import mask

os.environ.setdefault("PYTHONPATH", "tensorflow/models/research:tensorflow/models/research/slim")
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

class CocoConvertRecord():
    @staticmethod
    def _create_tf_record_from_coco_annotations(coco_json_file,image_folder_path,save_tf_record_file_path):
        if True:
            with tf.gfile.GFile(coco_json_file, 'rb') as fid:
                # print("fid",fid)
                groundtruth_data = json.loads(fid.read().decode("utf-8"))
                images = groundtruth_data['images']
                category_index = label_map_util.create_category_index(
                    groundtruth_data['categories'])

                annotations_index = {}
                if 'annotations' in groundtruth_data:
                    tf.logging.info(
                        'Found groundtruth annotations. Building annotations index.')
                    for annotation in groundtruth_data['annotations']:
                        print("annotation",annotation.keys())
                        image_id = annotation['image_id']
                        # print("image",image_id)
                        if image_id not in annotations_index:
                            annotations_index[image_id] = []
                        annotations_index[image_id].append(annotation)
                missing_annotation_count = 0
                for image in images:
                    image_id = image['id']
                    if image_id not in annotations_index:
                        missing_annotation_count += 1
                        annotations_index[image_id] = []
                tf.logging.info('%d images are missing annotations.',
                                missing_annotation_count)

                tf.logging.info('writing to output path: %s', save_tf_record_file_path)
                writer = tf.python_io.TFRecordWriter(save_tf_record_file_path)
                total_num_annotations_skipped = 0
                for idx, image in enumerate(images):
                    print("idx",idx)
                    annotations_list = annotations_index[image['id']]
                    # print("annotations_list",annotations_list,len(annotations_list))
                    _, tf_example, num_annotations_skipped = CocoConvertRecord.create_tf_example(
                        image, annotations_list, image_folder_path, category_index)
                    total_num_annotations_skipped += num_annotations_skipped
                    writer.write(tf_example.SerializeToString())
                writer.close()
                tf.logging.info('Finished writing, skipped %d annotations.',
                                total_num_annotations_skipped)
        # except Exception as ex:
        #     print("--- HATA (coco_convert_record/_create_tf_record_from_coco_annotations) ---", ex)

    @staticmethod
    def create_tf_example(image,annotations_list,image_dir,category_index):
        try:
            image_height = image['height']
            image_width = image['width']
            filename = image['file_name']
            image_id = image['id']

            # image_dir="/media/zeynep/data/segmentation_calisma/tensorflow_object_detection_create_coco_tfrecord/renkler"
            full_path = os.path.join(image_dir, filename)
            with tf.gfile.GFile(full_path, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            key = hashlib.sha256(encoded_jpg).hexdigest()

            xmin = []
            xmax = []
            ymin = []
            ymax = []
            is_crowd = []
            category_names = []
            category_ids = []
            area = []
            encoded_mask_png = []
            num_annotations_skipped = 0
            for object_annotations in annotations_list:
                (x, y, width, height) = tuple(object_annotations['bbox'])
                if width <= 0 or height <= 0:
                    num_annotations_skipped += 1
                    continue
                if x + width > image_width or y + height > image_height:
                    num_annotations_skipped += 1
                    continue
                xmin.append(float(x) / image_width)
                xmax.append(float(x + width) / image_width)
                ymin.append(float(y) / image_height)
                ymax.append(float(y + height) / image_height)
                is_crowd.append(object_annotations['iscrowd'])
                category_id = int(object_annotations['category_id'])
                category_ids.append(category_id)
                category_names.append(category_index[category_id]['name'].encode('utf8'))
                area.append(object_annotations['area'])
                # object_annotations['segmentation'] = np.array(object_annotations['segmentation'][0])

                run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                    image_height, image_width)
                binary_mask = mask.decode(run_len_encoding)
                if not object_annotations['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                pil_image = Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                encoded_mask_png.append(output_io.getvalue())

            feature_dict = {
                'image/height':
                    dataset_util.int64_feature(image_height),
                'image/width':
                    dataset_util.int64_feature(image_width),
                'image/filename':
                    dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id':
                    dataset_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256':
                    dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                    dataset_util.bytes_feature(encoded_jpg),
                'image/format':
                    dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                    dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                    dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                    dataset_util.float_list_feature(ymax),
                'image/object/class/label':
                    dataset_util.int64_list_feature(category_ids),
                'image/object/is_crowd':
                    dataset_util.int64_list_feature(is_crowd),
                'image/object/area':
                    dataset_util.float_list_feature(area),
            }
            # print("encoded_mask_png",encoded_mask_png)
            # if include_masks:
            feature_dict['image/object/mask'] = (
                dataset_util.bytes_list_feature(encoded_mask_png))
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            return key, example, num_annotations_skipped
        except:
            return False
class DatasetOperation:

    @staticmethod
    def create_coco_format_annotations_info(anndata,annotations_test_array,test_image_ids,annotations_train_array,train_image_ids):
        try:
            image_id = anndata['image_id']
            if image_id in test_image_ids:
                annotations_test_array.append(
                    {"id": anndata['id'],
                     "category_id": anndata['category_id'],
                     "image_id": image_id,
                     "segmentation": anndata['segmentation'],
                     'area': anndata['area'],  # o_width * o_height,
                     'iscrowd': anndata['iscrowd'],
                     'bbox': anndata['bbox']
                     }
                )
            elif image_id in train_image_ids:
                annotations_train_array.append(
                    {"id": anndata['id'],
                     "category_id": anndata['category_id'],
                     "image_id": image_id,
                     "segmentation": anndata['segmentation'],
                     'area': anndata['area'],  # o_width * o_height,
                     'iscrowd': anndata['iscrowd'],
                     'bbox': anndata['bbox']
                     }
                )
            # print("annotations_test_array",annotations_test_array)
            return annotations_test_array,annotations_train_array
        except Exception as ex:
            print("DatasetOperation.create_coco_format_images_info : {}".format(str(ex)))
            return None, None,

    @staticmethod
    def create_coco_format_images_info(info,test_img_folder, images_test_array, test_image_ids, train_img_folder, images_train_array, train_image_ids):
        try:
            if os.path.exists(os.path.join(test_img_folder, info['file_name'])):
                images_test_array.append(
                    {"id": info['id'],
                     "width": int(info['width']),
                     "height": int(info['height']),
                     "file_name": info['file_name']
                     }
                )
                test_image_ids.append(info['id'])
            elif  os.path.exists(os.path.join(train_img_folder, info['file_name'])):
                images_train_array.append(
                    {"id": info['id'],
                     "width": int(info['width']),
                     "height": int(info['height']),
                     "file_name": info['file_name']
                     }
                )
                train_image_ids.append(info['id'])
            return images_test_array,test_image_ids,images_train_array,train_image_ids
        except Exception as ex:
            print("DatasetOperation.create_coco_format_images_info : {}".format(str(ex)))
            return None, None,None,None

    @staticmethod
    def create_directory(directory_path):
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            return True
        except Exception as ex:
            print("DatasetOperation.create_tf_record_file_for_coco_format hatası : {}".format(str(ex)))
            return False

    #verilen dataset train ve test klasorlerine %80-%20 oranında ayriliyor. Harici evaluation veriseti verilmezse buda ayrıma dahil edilmelidir.
    @staticmethod
    def dataset_train_test_split_folder(dataset_path,save_path,train_rate=80):
        try:
            img_extensions = [".png",".jpg",".jpeg"]
            total_imgs = [os.path.join(dataset_path,img_name) for img_name in os.listdir(dataset_path) if os.path.splitext(img_name)[-1] in img_extensions]
            total_img_count = len(total_imgs)
            train_img_count = int((total_img_count*train_rate)/100)

            #train test klasorleri oluturuluyor
            save_train_dir_path = os.path.join(save_path,"train")
            save_test_dir_path = os.path.join(save_path, "test")
            result_train = DatasetOperation.create_directory(save_train_dir_path)
            if result_train:
                result_test = DatasetOperation.create_directory(save_test_dir_path)
                if result_test:
                    for index,file_path in enumerate(total_imgs):
                        filename = os.path.basename(file_path)
                        if index < train_img_count:
                            shutil.copy(file_path,os.path.join(save_train_dir_path,filename))
                        else:
                            shutil.copy(file_path,os.path.join(save_test_dir_path,filename))
                else:
                    print("DatasetOperation.dataset_train_test_split hatası test dosyası olusutulamiyor")
            else:
                print("DatasetOperation.dataset_train_test_split hatası train dosyası olusutulamiyor")

        except Exception as ex:
            print("DatasetOperation.create_tf_record_file_for_coco_format hatası : {}".format(str(ex)))

    #train ve test klasorleri icin coco formatinda json olusturma
    @staticmethod
    def coco_format_split_for_train_test_data(annotations_path,save_path):
        try:
            test_img_folder = os.path.join(save_path,"test")
            train_img_folder = os.path.join(save_path,"train")

            coco_test = {}
            coco_train = {}

            annotations_test = []
            annotations_train = []

            images_test = []
            images_train = []

            test_image_ids = []
            train_image_ids = []


            for file in os.listdir(annotations_path):
                filename, file_extension = os.path.splitext(file)
                if file_extension.lower() == '.json' and filename == 'annotation':
                    with open(os.path.join(annotations_path, file)) as f:
                        data = json.load(f)
                        f.close()
                    images_info = data["images"]
                    annotations_info = data["annotations"]
                    categories = data["categories"]

                    for imgdata in images_info:
                        images_test_array_result,test_image_ids_result,images_train_array_result,train_image_ids_result = DatasetOperation.create_coco_format_images_info(imgdata,test_img_folder,images_test,test_image_ids,train_img_folder,images_train, train_image_ids)
                        if (images_test_array_result is not None and test_image_ids_result is not None) and (images_train_array_result is not None and train_image_ids_result is not None):
                            images_test = images_test_array_result
                            test_image_ids = test_image_ids_result
                            images_train = images_train_array_result
                            train_image_ids = train_image_ids_result
                        else:
                            print("create_coco_format_images_info metodunda beklenmeyen bir hata images ile ilgili olustu.")

                    for anndata in annotations_info:
                        annotations_test_array_result, annotations_train_array_result = DatasetOperation.create_coco_format_annotations_info(
                            anndata, annotations_test, test_image_ids, annotations_train, train_image_ids)
                        if (annotations_test_array_result is not None  and annotations_train_array_result is not None):
                            annotations_test = annotations_test_array_result
                            annotations_train = annotations_train_array_result
                        else:
                            print("create_coco_format_images_info metodunda beklenmeyen bir hata annotation ile ilgili olustu.")

                    coco_test['annotations'] = annotations_test
                    coco_test['images'] = images_test
                    coco_test['categories'] = categories
                    print("annotation1", annotations_test[0].keys())

                    with open(os.path.join(save_path, "test.json"), 'w') as outfile:
                        json.dump(coco_test, outfile)
                        outfile.close()

                    coco_train['annotations'] = annotations_train
                    coco_train['images'] = images_train
                    coco_train['categories'] = categories

                    with open(os.path.join(save_path, "train.json"), 'w') as outfile:
                        json.dump(coco_train, outfile)
                        outfile.close()

                        return {'result': True, 'message': 'Tamamlandı.'}


        except Exception as ex:
            print("DatasetOperation.create_tf_record_file_for_coco_format hatası : {}".format(str(ex)))


    @staticmethod
    def create_tf_record_file_for_coco_format(coco_json_file,image_folder_path,save_tf_record_file_path):
        CocoConvertRecord._create_tf_record_from_coco_annotations(coco_json_file,image_folder_path,save_tf_record_file_path)
class Train:
    @staticmethod
    #model dosyasi: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.mdwg
    def train():
        command = "python " \
                  + os.path.join("object_detection", "legacy", "train.py") \
                  + " --gpuNums={} --train_dir={}/ --pipeline_config_path={}".format(0,"test/mask_rcnn_inception_v2_coco_2018_01_28",
                                                                                     "test/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config")

        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

DatasetOperation.dataset_train_test_split("test/dataset/images","test/split_dataset")
DatasetOperation.coco_format_annotation_control("test/dataset/annotations")
DatasetOperation.coco_format_split_for_train_test_data("test/dataset/annotations","test/testdataset/")
DatasetOperation.create_tf_record_file_for_coco_format("test/dataset/test.json","test/dataset/test","test/testdataset/test.record")
DatasetOperation.create_tf_record_file_for_coco_format("test/dataset/train.json","test/dataset/train","test/testdataset/train.record")




