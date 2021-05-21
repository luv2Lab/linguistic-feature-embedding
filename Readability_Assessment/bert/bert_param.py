# !/usr/bin/python3
# @Time : 2021/5/13 10:39
# @Author : ChenHanWu
# @File : bert_param.py
# @Software : PyCharm
# @Description : 存放提取bert向量的过程的参数

max_len = 512
task = "OneStop"
root_path = "../bert_json/"
BERT_BASE_DIR = '../multilingual_L-12_H-768_A-12'  # multilingual_L-12_H-768_A-12  对应的bert模型的主目录
INPUT_FILE = '../data/out_data/format_' + task + '.txt'  # 输入是一个每一条数据放在一行的文本
OUTPUT_FILE = root_path + task + '_' + str(max_len) + '.json'  # 输出的是一个bert装换后的json文件，需要进一步解析取出向量
input_file = INPUT_FILE
vocab_file = BERT_BASE_DIR + '/vocab.txt'
bert_config_file = BERT_BASE_DIR + '/bert_config.json'
init_checkpoint = BERT_BASE_DIR + '/bert_model.ckpt'
layers = "-1"  # 提取的最后一层
batch_size = 4
