import sys
import os
import os.path as osp
import argparse

def getTrainingDataList(data_home, useVal=True):
    train_home = osp.join(data_home, 'train')
    train_paths = os.listdir(train_home)

    val_home = osp.join(data_home, 'valid')
    val_paths = os.listdir(val_home)

    all_img_path = []
    all_lbl_path = []

    for pd in train_paths:
        img_dir = osp.join(train_home, pd, 'Images')
        img_paths = os.listdir(img_dir)
        for img_p in img_paths:
            img_path = osp.abspath(osp.join(img_dir, img_p))
            label_path = osp.abspath(img_path.replace('Images', 'Labels'))
            assert osp.exists(img_path)
            assert osp.exists(label_path)
            all_img_path.append(img_path)
            all_lbl_path.append(label_path)

    if useVal:
        for pd in val_paths:
            img_dir = osp.join(val_home, pd, 'Images')
            img_paths = os.listdir(img_dir)
            for img_p in img_paths:
                img_path = osp.abspath(osp.join(img_dir, img_p))
                label_path = osp.abspath(img_path.replace('Images', 'Labels'))
                assert osp.exists(img_path)
                assert osp.exists(label_path)
                all_img_path.append(img_path)
                all_lbl_path.append(label_path)

    with open('./fileListTrain.txt', 'w') as f:
        for i, l in zip(all_img_path, all_lbl_path):
            f.write(i + ' ' + l + '\n')

def getTestingDataList(data_home, target_dir):
    test_home = osp.join(data_home, 'test')
    test_paths = os.listdir(test_home)

    all_img_path = []
    all_pred_path = []
    for pd in test_paths:
        img_dir = osp.join(test_home, pd, 'Images')
        img_paths = os.listdir(img_dir)
        pred_dir = osp.join(target_dir, pd, 'Labels')
        for img_p in img_paths:
            img_path = osp.abspath(osp.join(img_dir, img_p))
            pred_path = osp.abspath(osp.join(pred_dir, img_p))
            assert osp.exists(img_path)
            all_img_path.append(img_path)
            all_pred_path.append(pred_path)

    with open('./fileListPred.txt', 'w') as f:
        for i, l in zip(all_img_path, all_pred_path):
            f.write(i + ' ' + l + '\n')

def parse_args(description='Prepare file lists.'):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('-u', help='UAVid home directory', type=str, default = './data/uavid')
    parser.add_argument('-p', help='prediction home directory', type=str, default='./output/pred')
    parser.add_argument('-v', help='Add validation set to fileListTrain.txt', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_home = args.u
    pred_home = args.p
    assert osp.isdir(data_home)
    getTrainingDataList(data_home, useVal = args.v)
    getTestingDataList(data_home, pred_home)