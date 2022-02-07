#!/bin/bash

MODEL_KIND='preincep'
IMG_FILE='/Users/akshaykekuda/Desktop/Jobs-OAs/Samsung/image-colorization_d6a566/test_images/442.jpg'
MODEL_PT='colorizer_preincep.model'
IMG_TYPE='gray'
python inference.py --input $IMG_FILE --image_type $IMG_TYPE --model $MODEL_KIND --model_pt $MODEL_PT
