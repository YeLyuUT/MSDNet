import numpy as np
from LLayer import *
from LLoss import *
from LMetric import *
from LNet import *
from LSummary import *
from LSolverOptions import *
import LDataInput
import sys
import multiprocessing
from model.FCN8sMultiScale import*
import PIL
from toolkit.UAVColorEncoder import UAVImageColorEncoder
import os
import os.path as osp
import argparse

##### Model config. #####
H = 1024 # input image height.
W = 1024 # input image width.
ClassNum = 8 # output class number.
loss_weights = [0.,0.,3.] # loss weights for the three streams. From coarse scale to fine scale.

##### Solver. #####
##### Train. #####
solver_opt_train = solverOptions()
solver_opt_train.phase = 'Train'
solver_opt_train.solver_type = 'Adam' # solver type. 'Adam', others like 'RMSProp' can be used as well.
solver_opt_train.gamma = 0.96 # learning rate decay factor.
solver_opt_train.base_lr = 1e-5 # start learning rate.
solver_opt_train.weight_decay = 1e-5 # weight decay.
solver_opt_train.snapshot = 4350 # number of iterations to save model weights.
solver_opt_train.batch_size = 1 # batch size of 1, input larger image crop instead of larger batch.
solver_opt_train.epoch = 40 # training epochs.
solver_opt_train.stepsize = 4350 * solver_opt_train.epoch / 55 # learning rate exponentially scale to 0.1*base_lr
solver_opt_train.snapshow_prefix = './checkpoints'
solver_opt_train.summary_dir = os.path.join(solver_opt_train.snapshow_prefix, 'summary')
###### Test. ######
solver_opt_test = solverOptions()
solver_opt_test.phase = 'Predict'
solver_opt_test.snapshow_prefix = './checkpoints'
solver_opt_test.summary_dir = os.path.join(solver_opt_test.snapshow_prefix, 'summary')

###### Define Network. #####
TheNet = FCN8sMultiScale(H,W,ClassNum, train=True)

def trace_back():
  exc_type, exc_value, exc_traceback = sys.exc_info()
  traceback_details = {
    'filename': exc_traceback.tb_frame.f_code.co_filename,
    'lineno': exc_traceback.tb_lineno,
    'name': exc_traceback.tb_frame.f_code.co_name,
    'type': exc_type.__name__,
    'message': exc_value,  # or see traceback._some_str()
  }
  print(traceback_details)
  for p in multiprocessing.active_children():
    p.terminate()

def train_net(fileListPath,solver_opt,use_restore=False,restore_path=''):
  try:
    net = TheNet
    with net.Graph.as_default():
      Input_X = net.get_Input()
      logits = net.get_Logits()
      net.build_y_true_list()
      #define loss and metric
      loss = multiscale_sparse_softmax_cross_entropy_with_logits(y_logits_list=logits,y_true_list=net.get_y_true_list(),class_num=ClassNum,loss_weights=loss_weights,ignore_label=255,loss_weight=1.0)
      metric = accuracy_multiscale(net.get_Prediction_list(),net.get_y_true_list())
      net.Compile(loss,metric,solver_opt, restore_params = use_restore, restorePath=restore_path)
      fileList = LDataInput.getPairFileLists(fileListPath)
      gen = LDataInput.DataGenerator(fileList=fileList,output_channels=ClassNum,batch_size=solver_opt.batch_size,epoch=solver_opt.epoch,shuffle = True)
      #gen.set_data_augmentation_func(net.filter_with_data_augmentation_pipeline)
      gen.set_label_encoder(UAVImageColorEncoder())
      net.fit_generator(gen)
  except Exception as e:
    trace_back()

def train_net_with_pretrain_model(fileListPath,solver_opt,pretrain_model_restore_path):
  try:
    net = TheNet
    with net.Graph.as_default():
      Input_X = net.get_Input()
      logits = net.get_Logits()
      net.build_y_true_list()
      
      loss = multiscale_sparse_softmax_cross_entropy_with_logits(y_logits_list=logits,y_true_list=net.get_y_true_list(),class_num=ClassNum,loss_weights=loss_weights,ignore_label=255,loss_weight=1.0,loss_focus=False)
      metric = accuracy_multiscale(net.get_Prediction_list(),net.get_y_true_list())

      net.CompileWithPretrainModel(loss,metric,solver_opt,pretrain_model_restore_path)
      fileList = LDataInput.getPairFileLists(fileListPath)
      gen = LDataInput.DataGenerator(fileList=fileList,output_channels=ClassNum,batch_size=solver_opt.batch_size,epoch=solver_opt.epoch,shuffle = True)
      #gen.set_data_augmentation_func(net.filter_with_data_augmentation_pipeline)
      gen.set_label_encoder(UAVImageColorEncoder())
      net.fit_generator(gen)
  except Exception as e:
    trace_back()

def predict_net_for_big_image(fileListPath,item_num_per_line_in_txt,solver_opt,save_dir_path,restorePath='',output_colorLabel=False):
  try:
    net = TheNet
    clr_encoder = UAVImageColorEncoder()
    with net.Graph.as_default():
      Input_X = net.get_Input()
      logits = net.get_Logits()
      net.Compile(None,None,solver_opt,restore_params = True,
        restorePath=restorePath)

      if item_num_per_line_in_txt==1:
        fileList = LDataInput.getSingleFileLists(fileListPath)
      if item_num_per_line_in_txt==2:
        fileList,predList = LDataInput.getDoubleFileLists(fileListPath)
      if item_num_per_line_in_txt==3:
        fileList,groundTruth,predList = LDataInput.getTripleFileLists(fileListPath)

      gen = LDataInput.DataGenerator(fileList=fileList,output_channels=ClassNum,batch_size=1,epoch=1,shuffle = False)

      imgsize=1024
      sp_ref=768

      print('Begin Prediction...')
      idx = 0
      for img in gen.generateImg():
        w = img.shape[2]
        h = img.shape[1]
        print("w,h:",w,h)
        n_w = (w-1)//sp_ref+1
        n_h = (h-1)//sp_ref+1
        print('number_sub_img:%i'%(n_w*n_h))
        if n_w>1:
          sp_w = (w-imgsize)//(n_w-1)
        else:
          sp_w=0
        if n_h>1:
          sp_h = (h-imgsize)//(n_h-1)
        else:
          sp_h=0
        out = np.zeros([h,w,ClassNum],dtype = np.float)
        out_count = np.zeros([h,w],dtype = np.float)
        for i in range(n_h):
          for j in range(n_w):
            if i!=n_h-1 and j!=n_w-1:
              _img = img[:,sp_h*i:sp_h*i+imgsize,sp_w*j:sp_w*j+imgsize]
              out[sp_h*i:sp_h*i+imgsize,sp_w*j:sp_w*j+imgsize,:] += np.squeeze(net.calcUnary(_img))
              out_count[sp_h*i:sp_h*i+imgsize,sp_w*j:sp_w*j+imgsize] += 1
            elif i!=n_h-1 and j==n_w-1:
              _img = img[:,sp_h*i:sp_h*i+imgsize,w-imgsize:w]
              out[sp_h*i:sp_h*i+imgsize,w-imgsize:w,:] += np.squeeze(net.calcUnary(_img))
              out_count[sp_h*i:sp_h*i+imgsize,w-imgsize:w] += 1
            elif i==n_h-1 and j!=n_w-1:
              _img = img[:,h-imgsize:h,sp_w*j:sp_w*j+imgsize]
              out[h-imgsize:h,sp_w*j:sp_w*j+imgsize,:]+=np.squeeze(net.calcUnary(_img))
              out_count[h-imgsize:h,sp_w*j:sp_w*j+imgsize] += 1
            else:
              _img = img[:,h-imgsize:h,w-imgsize:w]
              out[h-imgsize:h,w-imgsize:w,:]+=np.squeeze(net.calcUnary(_img))
              out_count[h-imgsize:h,w-imgsize:w] += 1

        outImg = np.argmax(out,axis=-1)
        if output_colorLabel:
          outImg = clr_encoder.inverse_transform(outImg)

        if item_num_per_line_in_txt==1:
          bname = os.path.basename(fileList[idx])
          checkCreateDirectory(save_dir_path)
          filepathname = os.path.join(save_dir_path,bname)
          print('save:',filepathname)
          PIL.Image.fromarray(outImg.astype(np.uint8)).save(filepathname)
        if item_num_per_line_in_txt==2:
          filepathname = predList[idx]
          print('save:',filepathname)
          checkCreateDirectory(osp.dirname(filepathname))
          PIL.Image.fromarray(outImg.astype(np.uint8)).save(filepathname)
        if item_num_per_line_in_txt==3:
          filepathname = predList[idx]
          print('save:',filepathname)
          checkCreateDirectory(osp.dirname(filepathname))
          PIL.Image.fromarray(outImg.astype(np.uint8)).save(filepathname)
        idx+=1
      print('Prediction finished.')

  except Exception as e:
    trace_back()

def checkCreateDirectory(dirpath):
  if not osp.isdir(dirpath):
    os.makedirs(dirpath)
  if not osp.isdir(dirpath):
    raise ValueError('Cannot create directory:%s'%(dirpath))
  return True

def train(fileListPath,restore_path=None):
  solver_opt = solver_opt_train
  checkCreateDirectory(solver_opt.snapshow_prefix)
  solver_opt.summary_dir = os.path.join(solver_opt.snapshow_prefix,'summary')
  if restore_path:
    train_net(fileListPath, solver_opt, use_restore=True, restore_path=restore_path)
  else:
    train_net(fileListPath, solver_opt, use_restore=False, restore_path=restore_path)

def train_with_pretrain(fileListPath, restore_path):
  assert restore_path is not None, 'restore_path is empty'
  solver_opt = solver_opt_train
  checkCreateDirectory(solver_opt.snapshow_prefix)
  solver_opt.summary_dir = os.path.join(solver_opt.snapshow_prefix,'summary')
  train_net_with_pretrain_model(fileListPath,solver_opt, pretrain_model_restore_path=restore_path)

def predict_for_big_image(fileListPath,restore_path,pred_dir = './output'):
  assert restore_path is not None, 'restore_path is empty'
  solver_opt = solver_opt_test
  predict_net_for_big_image(fileListPath,2,solver_opt,pred_dir,restore_path,output_colorLabel = True)

def parse_args(description='MSD Main'):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('-m',
                      help='type of main script to run. '
                           '"t" for train. '
                           '"tp for train with pre-trained model". '
                           '"p" for prediction',
                      choices=['t', 'tp', 'p'])
  parser.add_argument('-f', help='file list path', type=str, required=True)
  parser.add_argument('-w', help='model weights to load', type=str, default=None, required=False)
  args = parser.parse_args()
  return args

if __name__=='__main__':
  args = parse_args()
  if args.m=='t':
    train(args.f, args.w)
  elif args.m=='tp':
    train_with_pretrain(args.f, args.w)
  elif args.m=='p':
    predict_for_big_image(args.f, args.w)