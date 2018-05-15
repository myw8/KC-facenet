#!/bin/bash
help_info ()
{
	echo "****************************************************************************"
	echo "*"
	echo "* MODULE:             Linux Script - facenet"
	echo "*"
	echo "* COMPONENT:          This script used to run facenet python process"
	echo "*"
	echo "* REVISION:           $Revision: 1.0 $"
	echo "*"
	echo "* DATED:              $Date: 2018-04-09 15:16:28 +0000 () $"
	echo "*"
	echo "* AUTHOR:             PCT"
	echo "*"
	echo "***************************************************************************"
	echo ""
	echo "* Copyright yanhong.jia@kuang-chi.com. 2020. All rights reserved"
	echo "*"
	echo "***************************************************************************"
}

usage()
{
    echo "##################Usage################## "
    echo "You provided $# parameters,but 3 are required. \
    th first parameter is run cmd,\
    the second parameter is datasets path,\
    the third parameter is project path"
    #echo " facenet pretrain  use: ./facenet.sh pretrain "
    #echo "facenet train on own images: ./facenet.sh train "
    echo "examples are as follows:"
    #echo " facenet test on own images: ./facevalid.sh datapair /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image  /home/yiqi.liu-2/yanhong.jia/project/facevalid"
    #echo " facenet test on own images: ./facevalid.sh facenet_train /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image  /home/yiqi.liu-2/yanhong.jia/project/facevalid"
    echo " facenet test on own images: ./facevalid.sh faceKNN_train /home/yiqi.liu-4/yanhong.jia/datasets/face_image/SELFDATA  /home/yiqi.liu-4/yanhong.jia/project/KC-facenet"
    echo "facenet test on own images: ./facevalid.sh face_test  /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image  /home/yiqi.liu-2/yanhong.jia/project/facevalid"


}
case $1 in
    datapair|s)
        export CUDA_VISABLE_DEVICES=1
        python src/datapair.py --input_dir $2 \
        --output_dir $3 \
        --mode gen_pairs
        ;;
    facenet_valid)
        export PYTHONPATH=$3/src
        export CUDA_VISABLE_DEVICES=1

        python src/align/align_dataset_mtcnn.py  \
        $2/valid_name  \
        $2/valid_name_align_160  \
        --image_size 160 \
        --margin 32 \
        --random_order \
        --gpu_memory_fraction 0.2

        echo "my_validate_on_lfw ....."
        python src/my_validate_on_lfw.py \
        --lfw_dir $2/valid_name_align_160 \
        --model $3/models/20180428-181544 \
        --lfw_pairs $2/valid_name/pairs.txt \
        ;;
    facenet_train)
        export PYTHONPATH=$3/src
        export CUDA_VISABLE_DEVICES=2

        python src/align/align_dataset_mtcnn.py  \
        $2/train_name  \
        $2/train_name_160  \
        --image_size 160 \
        --margin 32 \
        --random_order \
        --gpu_memory_fraction 0.2
        echo "train_tripletloss ....."
        python src/train_tripletloss.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models \
          --pretrained_model ./models/20180504-215624/model-20180504-215624.ckpt-66002 \
          --data_dir $2/train_name_160 \
          --model_def models.inception_resnet_v1 \
          --optimizer RMSPROP \
          --learning_rate 0.01 \
          --weight_decay 1e-4 \
          --max_nrof_epochs 500  \
          --people_per_batch 15 \
          --images_per_person 10 \
          --gpu_memory_fraction 0.8
        ;;
    faceKNN_train)
        export PYTHONPATH=$3/src
        export CUDA_VISABLE_DEVICES=1

        echo "align hr dataset 128 -->160"
        python ./src/align/align_dataset_mtcnn.py \
        $2/valid_name  \
        $2/valid_name_align_160 \
        --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.8
        :'
        echo "train facenet ...."
        python src/myclassifier.py TRAIN \
            --data_dir \
            $2/train_name_align_160\
            --model \
            $3/models/20180428-181544\
            --classifier_filename \
            $3/models/classifier.pkl \
            --batch_size 100 \
            --min_nrof_images_per_class 5 \
            --nrof_train_images_per_class 5 \
            --use_split_dataset
        '
        python src/classifier.py TRAIN \
            --data_dir \
            $2/valid_name_align_160 \
            --model \
            $3/models/20180428-181544 \
            --classifier  KNN \
            --classifier_filename \
            $3/models/KNN.pkl \
            --batch_size 100
        ;;
    faceknn_test)
        export PYTHONPATH=$3/src
        export CUDA_VISABLE_DEVICES=1

        echo "align hr dataset -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/test_name  \
        $2/test_name_align_160 \
        --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.2

        echo "test facenet ...."
         python src/myclassifier.py CLASSIFY \
        --data_dir  $2/test_name_align_160 \
        --model $3/models/20180428-181544 \
        --classifier_filename $3/models/KNN.pkl  --batch_size 100
        ;;
    face_test)
        export PYTHONPATH=$3/facenet
        export CUDA_VISABLE_DEVICES=1
        echo "align valid_name_lr_32 dataset 32 -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/valid_name_lr_32  \
        $2/valid_name_lr_mtcnnpy_160 \
        --image_size 160 --margin 32 \
        --random_order --gpu_memory_fraction 0.8
        echo "lr to fsr ..................................."

        python FSRGANv1.0/test_FSRGAN.py  CLASSIFY \
           --lr_data_dir $2/valid_name_lr_32 \
           --lr_bicu_dir $2/valid_name_lr_bicubic_128 \
           --sr_data_dir $2/valid_name_fsr_128
        echo "lr to sr ..................................."
        python  SRGANv1.0/test_SRGAN.py  CLASSIFY \
            --lr_data_dir $2/valid_name_lr_32 \
            --lr_bicu_dir $2/valid_name_lr_bicubic \
            --sr_data_dir $2/valid_name_sr_128



        echo "align sr image dataset 128 -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/valid_name_sr_128  \
        $2/valid_name_sr_mtcnnpy_160 \
        --image_size 160 --margin 32 \
        --random_order --gpu_memory_fraction 0.8

        echo "align fsr image dataset 128 -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/valid_name_fsr_128  \
        $2/valid_name_fsr_mtcnnpy_160 \
        --image_size 160 --margin 32 \
        --random_order --gpu_memory_fraction 0.8

        echo "val bucibuc face......"
        python src/myclassifier.py CLASSIFY \
        --data_dir  $2/valid_name_lr_mtcnnpy_160 \
        --model $3/models/20180419-140738 \
        --classifier_filename $3/models/classifier.pkl  --batch_size 100

        echo "val SR face......"
        python src/myclassifier.py CLASSIFY \
        --data_dir $2/valid_name_sr_mtcnnpy_160 \
        --model $3/models/20180419-140738 \
        --classifier_filename $3/models/classifier.pkl --batch_size 100

        echo "val FSR face......"
        python src/myclassifier.py CLASSIFY \
        --data_dir $2/valid_name_fsr_mtcnnpy_160 \
        --model $3/models/20180419-140738 \
        --classifier_filename $3/models/classifier.pkl --batch_size 100
        ;;
    *)
		help_info
	    usage
		exit 1
    ;;
esac
exit 0