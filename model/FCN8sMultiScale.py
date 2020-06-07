import tensorflow as tf
import numpy as np
from LLayer import *
from LLoss import *
from LMetric import *
from LNet import *

#H: height
#W: width
#ClassNum: number of predicted classes 
def FCN8sMultiScale(H,W,ClassNum,train=True):
  net = Net('FCN8sMultiScale')
  with net.Graph.as_default():
    if train:
        Input_X0 = net.StartNetDef([1,H,W,3])
        Input_X0 = tf.squeeze(Input_X0,axis=0)
        Input_X0 = net.build_data_augmentation_pipeline(Input_X0)
        Input_X0 = tf.expand_dims(Input_X0,axis=0)
    else:
        Input_X0 = net.StartNetDef([1,H,W,3])
    Input_X0 = Input_X0-0.5
    pad = [[0,0],[0,0]]
    #padding_list = [[[0,0],[0,0]],[[0,0],[0,0]]]
    padding_list = None
    Input_X1 = Space2Batch(Input_X0,paddings = pad,block_size=2,name='ImgScale1')
    Input_X2 = Space2Batch(Input_X1,paddings = pad,block_size=2,name='ImgScale2')
    Input_list = [Input_X2,Input_X1,Input_X0]

    X = Conv2DMultiScale(Input_list,ksize=3,stride=1,o_c=64,use_relu=True,concat=False,separate_towers=True,name='Conv1_1')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=64,use_relu=True,separate_towers=True,name='Conv1_2')
    Pool1 = MaxPooling2DMultiScale(X,ksize=2,stride=2,name='Pool1')
    #Pool1 = Conv2DMultiScale(X,ksize=4,stride=2,o_c=64,use_relu=True,separate_towers=True,name='Pool1')
    #2
    X = Conv2DMultiScale(Pool1,ksize=3,stride=1,o_c=128,use_relu=True,concat=False,separate_towers=True,name='Conv2_1')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=128,use_relu=True,separate_towers=True,name='Conv2_2')
    Pool2 = MaxPooling2DMultiScale(X,ksize=2,stride=2,name='Pool2')
    #Pool2 = Conv2DMultiScale(X,ksize=4,stride=2,o_c=128,use_relu=True,separate_towers=True,name='Pool2')
    #3
    #padding_list = [[[0,1],[0,0]],[[0,0],[0,0]]]
    X = Conv2DMultiScale(Pool2,ksize=3,stride=1,o_c=128,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='Conv3_1')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=128,use_relu=True,padding_list=padding_list,separate_towers=True,name='Conv3_2')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=128,use_relu=True,padding_list=padding_list,separate_towers=True,name='Conv3_3')
    Pool3 = MaxPooling2DMultiScale(X,ksize=2,stride=2,name='Pool3')
    #Pool3 = Conv2DMultiScale(X,ksize=4,stride=2,o_c=256,use_relu=True,separate_towers=True,name='Pool3')

    #4
    #padding_list = [[[0,0],[0,0]],[[0,1],[0,0]]]
    X = Conv2DMultiScale(Pool3,ksize=3,stride=1,o_c=256,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='Conv4_1')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=256,use_relu=True,padding_list=padding_list,separate_towers=True,name='Conv4_2')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=256,use_relu=True,padding_list=padding_list,separate_towers=True,name='Conv4_3')
    Pool4 = MaxPooling2DMultiScale(X,ksize=2,stride=2,name='Pool4')
    #Pool4 = Conv2DMultiScale(X,ksize=4,stride=2,o_c=512,use_relu=True,separate_towers=True,name='Pool4')
    
    #5
    X = Conv2DMultiScale(Pool4,ksize=3,stride=1,o_c=256,use_relu=True,concat=False,separate_towers=True,name='Conv5_1')
    X = Conv2DMultiScale(X,ksize=3,stride=1,o_c=256,use_relu=True,separate_towers=True,name='Conv5_2')
    X_Conv5 = Conv2DMultiScale(X,ksize=3,stride=1,o_c=256,use_relu=True,separate_towers=True,name='Conv5_3')

    #padding_list = [[[0,1],[0,0]],[[0,0],[0,0]]]
    X = Conv2DMultiScale(X_Conv5,ksize=3,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='FC6')
    #X = Conv2DMultiScale(X,ksize=1,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='FC6_2')
    #X = Conv2DMultiScale(X,ksize=1,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='FC6')
    
    #if train==True:
    #    X = DropoutMultiScale(X,keep_prob=0.5,name='Drop6')
    X = Conv2DMultiScale(X,ksize=1,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=True,separate_towers=False,name='FC7')
    #X = Conv2DMultiScale(X,ksize=1,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='FC7_1')
    #X = Conv2DMultiScale(X,ksize=1,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='FC7_2')
    #X = Conv2DMultiScale(X,ksize=1,stride=1,o_c=1024,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='FC7_3')
    
    net.pretrain_variables = net.get_Trainable_Variables()

    #Can drop if not enough memory
    if train==True:
        X = DropoutMultiScale(X,keep_prob=0.5,name='Drop7')
    
    score_fr = Conv2DMultiScale(X,ksize=1,stride=1,o_c=ClassNum,use_relu=True,padding_list=padding_list,concat=False,separate_towers=True,name='score_fr')
    #7
    #padding_list = [[[0,1],[0,0]],[[0,0],[0,0]]]
    X = Conv2DMultiScale(score_fr,ksize=3,stride=1,o_c=ClassNum,use_relu=False,separate_towers=True,name='score_1')
    score_pool4 = Conv2DMultiScale(Pool4,ksize=1,stride=1,o_c=ClassNum,use_relu=False,concat=False,separate_towers=True,name='score_pool4')
    fuse_pool4 = AddMultiScale(X,score_pool4,name='fuse_pool4')
    #8
    X = DeConv2DMultiScale(fuse_pool4,ksize=4,stride=2,o_c=ClassNum,
      top_sp_list=ShapeListForDeconv(Pool3,ClassNum),
      usebias=False,separate_towers=True,name='upscore_pool4')
    #padding_list = [[[0,0],[0,0]],[[0,1],[0,0]]]
    score_pool3 = Conv2DMultiScale(Pool3,ksize=1,stride=1,o_c=ClassNum,use_relu=False,padding_list=padding_list,concat=False,separate_towers=True,name='score_pool3')
    fuse_pool3 = AddMultiScale(X,score_pool3,name='fuse_pool3')
    #9
    #padding_list = [[[0,0],[0,0]],[[0,1],[0,0]]]
    upscore_32 = DeConv2DMultiScale(fuse_pool3,ksize=16,stride=8,o_c=ClassNum,
      top_sp_list=ShapeListForDeconv(Input_list,ClassNum),
      usebias=False,padding_list=padding_list,separate_towers=True,name='upscore_32')

    net.all_trainable_variables = net.get_Trainable_Variables()

    logits = net.EndNetDef(logits=upscore_32)
  return net
