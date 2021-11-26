import os
import sys
import glob
import datetime

from pixellib.custom_train import instance_custom_training
import tensorflow as tf

from bubble_predict import path_load, model_predict_maskrcnn, pickle_to_json


if __name__ == "__main__":

    # warning 메시지 x
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    ########################################################################################################################
    # I.Parameter 확인 및 설정
    # ----------------------------------------------------------------------------------------------------------------------
    # 1. 변수 선언
    # 2. SYSARGV에서 train_predict 식별
    # 3. 작업일시 생성
    # 4. 파일 path 지정
    ########################################################################################################################
    
    # 1. 변수 선언
    train_predict = None

    # 2. SYSARGV에서 train_predict 식별
    argv = ["program_id", "train_predict"]
    for i in range(len(sys.argv)):
        exec (argv[i] + "=" + "'" + sys.argv[i] + "'")
    
    # 3. 작업일시 생성
    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d")
    
    # 4. 파일 path 지정
    train_dataset_path, test_file_path, save_weight_path, image_save_path, pickle_save_path, json_save_path = path_load(date)
    
    ########################################################################################################################
    # II.기포인식 AI모델 학습 및 예측
    # ----------------------------------------------------------------------------------------------------------------------
    # 1. mask-rcnn 모델 학습
    #  1) 모델 파라미터 설정
    #  2) mask-rcnn 모델 정의
    #  3) pretrained 가중치 load
    #  4) 학습 데이터셋 load
    #  5) mask-rcnn 모델 학습
    # 2. mask-rcnn 모델 예측
    # 3. 예측 결과 값 후처리 진행
    ########################################################################################################################
    
    # 1. mask-rcnn 모델 학습
    # 시스템 입력 변수 train_predict가 train일 경우 학습 진행
    if train_predict == "train":
        
        # 1) 모델 파라미터 설정
        detection_threshold = 0.7
        num_classes = 1
        batch_size = 2
        num_epochs = 50
        max_gt_instances = 200
        layers = "all"

        # 2) mask-rcnn 모델 정의
        train_maskrcnn = instance_custom_training()
        train_maskrcnn.modelConfig(network_backbone = "resnet101", detection_threshold = detection_threshold, 
                                   num_classes = num_classes, batch_size = batch_size)
        train_maskrcnn.config.MAX_GT_INSTANCES = max_gt_instances
        
        # 3) pretrained 가중치 load
        weight_file = "{}/mask_rcnn_coco.h5".format(save_weight_path)
        train_maskrcnn.load_pretrained_model(weight_file)
        
        # 4) 학습 데이터셋 load
        train_maskrcnn.load_dataset(train_dataset_path)
        
        # 5) mask-rcnn 모델 학습
        train_maskrcnn.train_model(num_epochs = num_epochs, path_trained_models = save_weight_path,
                                   layers = layers, augmentation = True)
        print("Train Success")
    
    # 2. mask-rcnn 모델 예측
    detection_max_instances = 200
    model_predict_maskrcnn(test_file_path, save_weight_path, image_save_path, pickle_save_path, detection_max_instances)
    print("Predict Success")

    # 3. 예측 결과 값 후처리 진행
    pickle_to_json(pickle_save_path, json_save_path)
    print("Post-Processing Success")

