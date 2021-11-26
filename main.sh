#!/bin/bash -e

echo "bubble segmentation $1 시작" $1
/home/ubuntu/anaconda3/envs/bubble_1123/bin/python -u ./main.py $1

echo "bubble segmentation 종료"
