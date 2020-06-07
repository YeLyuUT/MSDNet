from PIL import Image
import os
import os.path as osp
import numpy as np
from UAVColorEncoder import UAVImageColorEncoder
clr_enc = UAVImageColorEncoder()

def splitImagesIntoN(imgList,lblList,size,ref_sp,save_dir):
  '''
  size: int or [cropSizeH,cropSizeW]
  ref_sp: reference space between each crop, int or [refH,refW]
  '''

  assert(len(imgList)==len(lblList))
  save_dir = os.path.abspath(save_dir)
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)
  assert osp.isdir(save_dir)
  if isinstance(size,list):
    size_h = size[0]
    size_w = size[1]
  else:
    size_h = size
    size_w = size

  if isinstance(ref_sp,list):
    ref_sp_h = ref_sp[0]
    ref_sp_w = ref_sp[1]
  else:
    ref_sp_h = ref_sp
    ref_sp_w = ref_sp

  with open(os.path.join('fileListTrainCropped.txt'), 'w') as f:
    idx = 0
    for imgPath, lblPath in zip(imgList,lblList):
      print('imgPath:', imgPath)
      prefix = '%06i'%(idx)
      idx+=1
      img = np.array(Image.open(imgPath))
      lbl = np.array(Image.open(lblPath))
      if lbl.ndim==3:
        lbl = clr_enc.transform(lbl, dtype = np.uint8)

      h = img.shape[0]
      w = img.shape[1]

      n_w = (w-1)//ref_sp_w+1
      n_h = (h-1)//ref_sp_h+1

      sp_w = (w-size_w)//(n_w-1)
      sp_h = (h-size_h)//(n_h-1)

      imgBaseName = os.path.basename(imgPath)
      imgName,imgExt = imgBaseName.split('.')
      lblBaseName = os.path.basename(lblPath)
      lblName,lblExt = lblBaseName.split('.')

      for i in range(n_h):
        for j in range(n_w):
          if i!=n_h-1 and j!=n_w-1:
            _img = img[sp_h*i:sp_h*i+size_h,sp_w*j:sp_w*j+size_w]
            _lbl = lbl[sp_h*i:sp_h*i+size_h,sp_w*j:sp_w*j+size_w]
            #print('1',[sp_h*i,sp_h*i+size_h,sp_w*j,sp_w*j+size_w])
          elif i!=n_h-1 and j==n_w-1:
            _img = img[sp_h*i:sp_h*i+size_h,w-size_w:w]
            _lbl = lbl[sp_h*i:sp_h*i+size_h,w-size_w:w]
            #print('2',[sp_h*i,sp_h*i+size_h,w-size_w,w])
          elif i==n_h-1 and j!=n_w-1:
            _img = img[h-size_h:h,sp_w*j:sp_w*j+size_w]
            _lbl = lbl[h-size_h:h,sp_w*j:sp_w*j+size_w]
            #print('3',[h-size_h,h,sp_w*j,sp_w*j+size_w])
          else:
            _img = img[h-size_h:h,w-size_w:w]
            _lbl = lbl[h-size_h:h,w-size_w:w]
            #print('4',[h-size_h,h,w-size_w,w])

          path_img = os.path.join(save_dir,prefix+'_%i_%i.'%(i,j)+imgExt)
          path_lbl = os.path.join(save_dir,prefix+'TrainId_%i_%i.'%(i,j)+lblExt)
          print(path_img+' '+path_lbl)
          Image.fromarray(_img).save(path_img)
          Image.fromarray(_lbl).save(path_lbl)
          f.write(path_img+' '+path_lbl+'\n')

