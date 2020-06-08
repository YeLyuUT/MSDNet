import tensorflow as tf
import copy
import os
import LLayer
from LDataInput import LOutOfRangeError
import LLoss
import LDataInput
from matplotlib import pyplot as plt
import numpy as np
import sys
import multiprocessing
import pickle
from LSummary import *

class Net():
  def __init__(self,name):
    self.Graph = tf.Graph()
    self.prediction = None
    self.Activate()

  def Activate(self):
    self.Graph.as_default()
    self.sess = tf.Session(graph = self.Graph)

  def get_Input(self):
    return self.Input

  def get_Input_Y(self):
    return self.Input_Y

  def get_Output(self):
    return self.Output

  def get_Model(self):
    return self.Model

  def get_Logits(self):
    return self.Logits

  def get_Prediction(self):
    return self.prediction

  def get_Prediction_list(self):
    return self.prediction_list

  def get_Unary(self):
    return self.unary

  def get_y_true_list(self):
    return self.y_true_list

  def build_y_true_list(self):
    print('building y_true_list.')
    self.y_true_list=[]
    y_true_0 = self.get_Input_Y()
    y_true_0 = tf.expand_dims(y_true_0,3)
    paddings=[[0,0],[0,0]]
    y_true_1 = LLayer.Space2Batch(y_true_0,paddings=paddings,block_size=2,name='y_true_1')
    y_true_2 = LLayer.Space2Batch(y_true_1,paddings=paddings,block_size=2,name='y_true_2')
    
    y_true_0 = tf.squeeze(y_true_0,axis=3)
    y_true_1 = tf.squeeze(y_true_1,axis=3)
    y_true_2 = tf.squeeze(y_true_2,axis=3)
    
    self.y_true_list.append(y_true_2)
    self.y_true_list.append(y_true_1)
    self.y_true_list.append(y_true_0)
    print('Done.')
    return self.y_true_list

  def build_data_augmentation_pipeline(self,data_aug_in):
    print('Use Color Augmentation')
    # Randomly adjust hue, contrast and saturation.
    image = tf.image.random_hue(data_aug_in, max_delta=0.03)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.2)
    # Some of these functions may overflow and result in pixel
    # values beyond the [0, 1] range. It is unclear from the
    # documentation of TensorFlow 0.10.0rc0 whether this is
    # intended. A simple solution is to limit the range.
    # Limit the image pixels between [0, 1] in case of overflow.
    image = tf.minimum(image, 1.0)
    data_aug_out = tf.maximum(image, 0.0)
    return data_aug_out

  def filter_with_data_augmentation_pipeline(self,image):
      return self.sess.run(self.data_aug_out,feed_dict={self.data_aug_in:image})

  def StartNetDef(self,shape_feature):
    print('Start Net Definition...')
    #shape_feature: [H,W,C]
    input_X = tf.placeholder(tf.float32,shape = shape_feature)
    input_Y = tf.placeholder(tf.int32,shape=shape_feature[:-1])
    self.Input = input_X
    self.Input_Y = input_Y
    return self.Input

  def EndNetDef(self,logits):
    self.Logits = logits
    print('End Net Definition.')
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,max_to_keep=None)
    self.__get_Predict()
    self.global_step_from_zero = self.__get_global_step(0)
    return self.Logits

  def __parse_solver_opts(self,solverOptions):
    self.solverOptions = copy.copy(solverOptions)
    self.global_step = self.global_step_from_zero+tf.constant(self.solverOptions.start_step,dtype=tf.int32)
    self.lr = self.__get_learning_rate(self.solverOptions)
    self.saverFilePath = self.__get_saver_file_path(self.solverOptions)
    self.__get_solver(self.solverOptions,self.lr)
    self.print_solver_opts()
    

  def print_solver_opts(self):
    if self.solverOptions is None:
      print('solver is not defined')
      return False
    else:
      opts = self.solverOptions
      print(opts)
    return True

  def __get_saver_file_path(self,solverOptions):
    saverFilePath = os.path.join(solverOptions.snapshow_prefix,"model.ckpt")
    return saverFilePath

  def __get_solver(self,solverOptions,learning_rate):
    if self.solverOptions.solver_type=='Adam':
      self.optimizer =  tf.train.AdamOptimizer(learning_rate)
    elif self.solverOptions.solver_type=='Momentum':
      self.optimizer =  tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
    elif self.solverOptions.solver_type=='RMSProp':
      self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
      print('Cannot find solver type '+self.solverOptions.solver_type)
      raise ValueError('solver type error')
    print('Use '+self.solverOptions.solver_type+' as solver')
    return self.optimizer

  def __get_global_step(self,start_step=0):
    with tf.name_scope("global_step"):
      global_step = tf.get_variable("global_step",initializer=tf.constant(start_step),dtype=tf.int32,trainable=False)
    #self.sess.run(global_step.initializer)
    return global_step

  def __get_learning_rate(self,solverOptions):
    starter_learning_rate = solverOptions.base_lr
    if solverOptions.lr_policy=='step':
      print('learning rate policy: step')
      learning_rate = tf.train.exponential_decay(
      starter_learning_rate, self.global_step,
      solverOptions.stepsize, solverOptions.gamma, 
      staircase=True)
      # Passing global_step to minimize() will increment it at each step.
    elif solverOptions.lr_policy=='fixed':
      learning_rate=starter_learning_rate
    else:
      raise ValueError('lr_policy is not correctly defined')
    return learning_rate

  def Compile(self,loss,metrics,solverOptions,restore_params = False,restorePath=''):
    print('Compile Network...')
    if solverOptions.phase=='Train':
      print('Compile for Train Network...')
      self.__parse_solver_opts(solverOptions)
      #add weight decay loss to loss
      wd_loss_collection = tf.get_collection(LLayer.L_weight_collection)
      wd_loss = LLoss.regularization_loss(solverOptions.weight_decay)
      #add summary
      addSummaryWeights(wd_loss_collection)
      addSummaryLoss(loss,'xent_loss')
      addSummaryLoss(wd_loss,'wd_loss')
      loss = tf.add(loss,wd_loss)
      addSummaryLoss(loss,'total_loss')
      if isinstance(metrics,list):
        addSummaryAccuracy(metrics[-1],'accuracy')
      else:
        addSummaryAccuracy(metrics,'accuracy')
      self.summary_op = tf.summary.merge_all()
      self.summary_writer = tf.summary.FileWriter(solverOptions.summary_dir, graph=self.sess.graph)
      init_global = tf.global_variables_initializer()
      self.sess.run(init_global)  
      print('Model Initialized...')

      if restore_params:
        if restorePath=='':
          self.saver.restore(self.sess, self.saverFilePath)
        else:
          self.saver.restore(self.sess, restorePath)
        print("Model restored...")

      self.loss = loss
      self.metrics= metrics
      assert(self.optimizer is not None)
      assert(self.loss is not None)
      assert(self.metrics is not None)

      print('Initialize params in optimizer if there is any...')
      tempVariables = set(tf.global_variables())
      self.train_op = self.optimizer.minimize(self.loss,self.global_step_from_zero)
      self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tempVariables))
      print('Initialize params in optimizer done.')
      print('Ready for training.')

    elif solverOptions.phase=='Predict':
      print('Compile for Predict Network...')
      assert(restore_params == True)
      self.__parse_solver_opts(solverOptions)
      self.unary = tf.nn.softmax(self.Logits[-1],dim=-1)
      init_global = tf.global_variables_initializer()
      self.sess.run(init_global)  
      print('Model Initialized...')
      if restore_params:
        if restorePath=='':
          self.saver.restore(self.sess, self.saverFilePath)
        else:
          self.saver.restore(self.sess, restorePath)
        print("Model restored...")

      print('Ready for predicting.')
    else:
      print('solverOptions.phase is not valid, please use "Train" or "Test" or "Predict''.')
      raise ValueError('solverOptions.phase is not valid')

  def CompileWithPretrainModel(self,loss,metrics,solverOptions,pretrain_model_restore_path):
    print('Compile Network With Pretrained Model...')
    assert(solverOptions.phase=='Train')
    self.__parse_solver_opts(solverOptions)
    #add weight decay loss to loss
    wd_loss_collection = tf.get_collection(LLayer.L_weight_collection)
    wd_loss = LLoss.regularization_loss(solverOptions.weight_decay)
    #add summary
    addSummaryWeights(wd_loss_collection)
    addSummaryLoss(loss,'xent_loss')
    addSummaryLoss(wd_loss,'wd_loss')
    loss = tf.add(loss,wd_loss)
    addSummaryLoss(loss,'total_loss')
    if isinstance(metrics,list):
      addSummaryAccuracy(metrics[-1],'accuracy')
    else:
      addSummaryAccuracy(metrics,'accuracy')
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(solverOptions.summary_dir, graph=self.sess.graph)
    init_global = tf.global_variables_initializer()
    self.sess.run(init_global) 
    print('Model Initialized...')

    print('Number of pretrain variables:',len(self.pretrain_variables))
    print('Number of finetune variables:',len(self.all_trainable_variables)-len(self.pretrain_variables))
    preTrainSaver = tf.train.Saver(self.pretrain_variables)
    preTrainSaver.restore(self.sess, pretrain_model_restore_path)
    print("Pretrain Model restored...")

    self.loss = loss
    self.metrics= metrics
    assert(self.optimizer is not None)
    assert(self.loss is not None)
    assert(self.metrics is not None)

    print('Initialize params in optimizer if there is any...')
    tempVariables = set(tf.global_variables())
    self.train_op = self.optimizer.minimize(self.loss,self.global_step_from_zero)
    self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tempVariables))
    print('Initialize params in optimizer done.')
    print('Ready for training.')

  def fit(self,features,labels):
    init_local = tf.local_variable_initializer()
    self.sess.run(init_local)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord = coord)
    try:
      while not coord.should_stop():
        batchNum = self.sess.run(self.global_step)
        rtvs = self.sess.run([self.loss,self.metrics,self.lr,self.train_op],feed_dict={self.Input:features,self.Input_Y:labels})
        print("batch:%d, loss:%f,accuracy:%f"%(batchNum,rtvs[0],rtvs[1]))
        self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)
    except tf.errors.OutOfRangeError:
      print('Done training -- batch limit reached')

    finally:
      coord.request_stop()
      coord.join(threads)
      self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)

  def fit_generator(self,generator):
    print('Fitting Sample Generator...')
    loss_list = list()
    accuracy_list = list()
    while(True):
      try:
        for X,Y in generator.generate():
          batchNum = self.sess.run(self.global_step)
          #print('BatchNum:',batchNum)
          #print('X.shape:',X.shape)
          #print('X.type:',X.dtype)
          #print('Y.shape:',Y.shape)
          #print('Y.type:',Y.dtype)
          rtvs = self.sess.run([self.loss,self.metrics,self.lr,self.train_op],feed_dict={self.Input:X,self.Input_Y:Y})
          print("batch:%d, loss:%f,lr:%f,accuracy:"%(batchNum,rtvs[0],rtvs[2]),rtvs[1])
          loss_list.append(rtvs[0])
          acc = rtvs[1]
          if isinstance(acc,list):
            accuracy_list.append(acc[-1])
          else:
            accuracy_list.append(acc)
          if batchNum%self.solverOptions.snapshot==0:
            summary_str = self.sess.run(self.summary_op,feed_dict={self.Input:X,self.Input_Y:Y})
            self.summary_writer.add_summary(summary_str,global_step = batchNum)
            self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)
      except LOutOfRangeError:
        print('Done training -- batch limit reached')
        dumpPath = os.path.join(self.solverOptions.summary_dir,'loss_accuracy.pickle')
        with open(dumpPath,'wb') as f:
          print('dump loss and accuracy to:%s'%(dumpPath))
          pickle.dump([loss_list,accuracy_list],f,pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print("Graph Running Error!")
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
         'filename': exc_traceback.tb_frame.f_code.co_filename,
         'lineno'  : exc_traceback.tb_lineno,
         'name'    : exc_traceback.tb_frame.f_code.co_name,
         'type'    : exc_type.__name__,
         'message' : exc_value, # or see traceback._some_str()
        }
        print(traceback_details)
        print('ExceptHook, terminate all child processes!')
        for p in multiprocessing.active_children():
           p.terminate()
      finally:
        self.saver.save(self.sess,self.saverFilePath)
        break

  def __get_Predict(self):
    if self.Logits is None:
      raise ValueError('Logits is not defined yet.')
    elif isinstance(self.Logits,list):
      print('build prediction list and prediction.')
      self.prediction_list = []
      for idx in range(len(self.Logits)):
        pred = tf.argmax(self.Logits[idx],axis=3,output_type=tf.int32,name='Prediction_%i'%(idx))
        self.prediction_list.append(pred)
      self.prediction = tf.argmax(self.Logits[-1], axis=3,output_type=tf.int32, name="Prediction")
    else:
      print('build prediction.')
      self.prediction = tf.argmax(self.Logits, axis=3,output_type=tf.int32, name="Prediction")

  def predict(self,X):
    rtvs = self.sess.run(self.prediction,feed_dict={self.Input:X})
    return rtvs

  def get_Trainable_Variables(self):
   return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  def build4WayUnary(self):
    self.unaries = LLayer.Space2Batch(tf.nn.softmax(self.Logits,dim=-1),block_size=2)

  def calc4WayUnary(self,X):
    rtvs = self.sess.run(self.unaries,feed_dict={self.Input:X})
    return rtvs

  def calcUnary(self,X):
    rtvs = self.sess.run(self.unary,feed_dict={self.Input:X})
    return rtvs

  def decodeLabel(self,predict_label):
    pass

  def saveImage(self,data,filepath):
    pass

