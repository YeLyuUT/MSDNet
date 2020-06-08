from CropImageToSize import splitImagesIntoN
import argparse

def parse_args(description='Prepare file lists.'):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('-u', help='UAVid home directory', type=str, default = './data/uavid')
    parser.add_argument('-p', help='prediction home directory', type=str, default='./output_pred')
    parser.add_argument('-v', help='Add validation set to fileListTrain.txt', action='store_true')
    args = parser.parse_args()
    return args

if __name__=='__main__':
	save_dir = './data/ImageLabelCropped'
	imgList=[]
	lblList=[]
	with open('./fileListTrain.txt','r') as f:
		lines = f.readlines()
		for line in lines:
			imgPath,lblPath = line.split()
			imgList.append(imgPath)
			lblList.append(lblPath)
	print(imgList, lblList)
	splitImagesIntoN(imgList,lblList,size=[1024,1024],ref_sp=[768,768],save_dir = save_dir)