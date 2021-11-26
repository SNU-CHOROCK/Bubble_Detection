import os
import glob
import json
import pickle
import datetime

from imantics import Polygons, Mask
import numpy as np
import cv2
import pixellib
from pixellib.instance import custom_segmentation
import tensorflow as tf

from util.get_path import PATH_DICT


def get_mask_point(masks):
    contain_val = []

    for a in range(masks.shape[2]):
        m = masks[:,:,a]
        mask_values = Mask(m).polygons()
        val = mask_values.points
        contain_val.append(val)
    contain_val = np.asarray(contain_val, dtype = object)

    return contain_val


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_box_instances(image, boxes, masks, class_ids, class_name, scores, text_size, box_thickness, text_thickness):

    n_instances = boxes.shape[0]
    N = boxes.shape[0]
    colors = [(0.0, 1.0, 1.0)] * N

    txt_color = (255,255,255)
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_name[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        color_rec = [int(c) for c in np.array(colors[i]) * 255]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rec, box_thickness)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size,  txt_color, text_thickness)

    return image


def path_load(date):
    
    ########################################################################################################################
    # 파일 path 지정
    # ----------------------------------------------------------------------------------------------------------------------
    # 1. 파일 경로 설정
    # 2. 파일 경로 폴더 생성
    ########################################################################################################################
    
    # 1. 파일 경로 설정
    train_dataset_path = PATH_DICT["train_dataset_path"]
    test_file_path = PATH_DICT["test_file_path"]
    save_weight_path = PATH_DICT["save_weight_path"]
    image_save_path = "{}_{}".format(PATH_DICT["image_save_path"], date)
    pickle_save_path = "{}_{}".format(PATH_DICT["pickle_save_path"], date)
    json_save_path = "{}_{}".format(PATH_DICT["json_save_path"], date)

    # 2. 파일 경로 폴더 생성
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(pickle_save_path, exist_ok=True)
    os.makedirs(json_save_path, exist_ok=True)
    
    return train_dataset_path, test_file_path, save_weight_path, image_save_path, pickle_save_path, json_save_path


def model_predict_maskrcnn(test_file_path, save_weight_path, image_save_path, pickle_save_path, detection_max_instances):

    ########################################################################################################################
    # mask-rcnn 모델 예측
    # ----------------------------------------------------------------------------------------------------------------------
    # 1. test 이미지 파일 리스트 load
    # 2. 예측 결과 저장 폴더 생성
    #  1) folder name 리스트 load
    #  2) 저장 폴더 생성(image, pickle)
    # 3. mask-rcnn 모델 예측
    #  1) mask-rcnn 모델 정의
    #  2) 학습 가중치 load
    #  3) mask-rcnn 모델 예측
    #  4) 예측 결과 저장
    ########################################################################################################################
    
    # 1. test 이미지 파일 리스트 load
    print("Test dataset load")
    
    test_file_list = []

    for path, dirs, files in os.walk(test_file_path):
        path = path.replace("\\", "/")
        for file in files:
            if os.path.splitext(file)[1].lower() == ".jpg" or os.path.splitext(file)[1].lower() == ".tif" or \
            os.path.splitext(file)[1].lower() == ".bmp":
                test_file_list.append(path + "/" + file)
    test_file_list.sort()
    
    test_file_count = len(test_file_list)
    print("Test {} images".format(test_file_count))

    # 2. 예측 결과 저장 폴더 생성
    # 1) folder name 리스트 load
    unique_folder_list = list(set([test_file.split("/")[-2] for test_file in test_file_list]))

    # 2) 저장 폴더 생성(image, pickle)
    for unique_folder in unique_folder_list:
        # image folder
        os.makedirs("{}/{}".format(image_save_path, unique_folder), exist_ok=True)
        # pickle folder
        os.makedirs("{}/{}".format(pickle_save_path, unique_folder), exist_ok=True)
    
    # 3. mask-rcnn 모델 예측
    # 1) mask-rcnn 모델 정의
    class_names = ["BG", "bubble"]
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes = 1, class_names = class_names)
    segment_image.DETECTION_MAX_INSTANCES = detection_max_instances

    # 2) 학습 가중치 load
    weight_list = sorted(glob.glob("{}/*.h5".format(save_weight_path)), key=os.path.getctime)
    weight_file = weight_list[-1]
    segment_image.load_model(weight_file)

    # 3) mask-rcnn 모델 예측
    for test_file in test_file_list:
        save_name, exe = test_file.split("/")[-1].split(".")
        folder_num = test_file.split("/")[-2]

        pickle_tmp = segment_image.segmentImage(test_file, show_bboxes=True, mask_points_values=False)
        
        output_image_name="{}/{}/{}.{}".format(image_save_path, folder_num, save_name, exe)
        
        masks = pickle_tmp[0]["masks"]
        contain_val = get_mask_point(masks)
        
        pickle_tmp[0]["file_name"] = "{}.{}".format(save_name, exe)
        pickle_tmp[0]["imageHeight"] = pickle_tmp[1].shape[0]
        pickle_tmp[0]["imageWidth"] = pickle_tmp[1].shape[1]
        pickle_tmp[0]["masks"] = contain_val
        
        image = cv2.imread(test_file)
        output_image = display_box_instances(image, pickle_tmp[0]["rois"], masks, pickle_tmp[0]["class_ids"], class_names,
                                             pickle_tmp[0]["scores"], text_size = 0.6, box_thickness = 2, 
                                             text_thickness = 0)
        cv2.imwrite(output_image_name, output_image)
        print("{} save complete".format(output_image_name))

        # 4) 예측 결과 저장
        with open("{}/{}/{}.pickle".format(pickle_save_path, folder_num, save_name), "wb") as fw:
            pickle.dump(pickle_tmp[0], fw)

            
def pickle_to_json(pickle_save_path, json_save_path):
    
    ########################################################################################################################
    # 예측 결과 값 후처리 진행
    # ----------------------------------------------------------------------------------------------------------------------
    # 1. 예측 결과 파일(pickle) 리스트 load
    # 2. 후처리 결과 저장 폴더 생성
    #  1) folder name 리스트 load
    #  2) 저장 폴더 생성(json)
    # 3. 예측 결과 데이터(pickle)에서 mark값을 추출한 후에 json으로 저장
    #  1) pickle 데이터 load
    #  2) json 형식에 맞게 데이터 후처리 진행
    #  3) 후처리 데이터 json 저장
    ########################################################################################################################
    
    # 1. 예측 결과 파일(pickle) 리스트 load
    pickle_file_list = []

    for path, dirs, files in os.walk(pickle_save_path):
        path = path.replace("\\", "/")
        for file in files:
            if os.path.splitext(file)[1].lower() == ".pickle":
                pickle_file_list.append(path + "/" + file)
    pickle_file_list.sort()
    
    # 2. 후처리 결과 저장 폴더 생성
    # 1) folder name 리스트 load
    unique_folder_list = list(set([pickle_file.split("/")[-2] for pickle_file in pickle_file_list]))
    
    # 2) 저장 폴더 생성(json)
    for unique_folder in unique_folder_list:
        os.makedirs("{}/{}".format(json_save_path, unique_folder), exist_ok=True)

    # 3. 예측 결과 데이터(pickle)에서 mark값을 추출한 후에 json으로 저장
    for pickle_file in pickle_file_list:
        
        # 1) pickle 데이터 load
        with open('{}'.format(pickle_file), 'rb') as fr:
            pickle_data = pickle.load(fr)

        # 2) json 형식에 맞게 데이터 후처리 진행
        shape_list = []

        for p_idx in range(pickle_data["masks"].shape[0]):
            mask_tmp = pickle_data["masks"][p_idx][0].tolist()

            shape_dict = {"label": "bubble"}
            shape_dict["points"] = mask_tmp
            shape_dict["group_id"] =  None
            shape_dict["shape_type"] = "polygon"
            shape_dict["flags"] = {}

            shape_list.append(shape_dict)

        file_name = pickle_data["file_name"]
        save_name = file_name.split(".")[0]
        folder_num = pickle_file.split("/")[-2]

        pickle_json_dict = {"version": "4.5.9", "flags": {}}
        pickle_json_dict["shapes"] = shape_list
        pickle_json_dict["imagePath"] = file_name
        pickle_json_dict["imageData"] = None
        pickle_json_dict["imageHeight"] = pickle_data["imageHeight"]
        pickle_json_dict["imageWidth"] = pickle_data["imageWidth"]

        # 3) 후처리 데이터 json 저장
        with open('{}/{}/{}.json'.format(json_save_path, folder_num, save_name), 'w', encoding='utf-8') as make_file:
            json.dump(pickle_json_dict, make_file, indent="\t")
            

if __name__ == "__main__":
    
    # 작업일시 생성
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d")
    
    # 파일 path 지정
    train_dataset_path, test_file_path, save_weight_path, image_save_path, pickle_save_path, json_save_path = path_load(date)

    # mask-rcnn 모델 예측
    model_predict_maskrcnn(test_file_path, save_weight_path, image_save_path, pickle_save_path, detection_max_instances)

    # 예측 결과 값 후처리 진행
    pickle_to_json(pickle_save_path, json_save_path)
