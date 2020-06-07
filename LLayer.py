import tensorflow as tf
import numpy as np
import scipy.stats as stats

L_weight_collection = 'L_weight_collection'
L_weight_decay_loss_collection = 'L_weight_decay_loss_collection'

def InitializerConstantType(val=0,dtype=tf.float32):
    return tf.constant_initializer(value=val,dtype=dtype)
def InitializerXavierType(uniform = False):
    return tf.contrib.layers.xavier_initializer(dtype=tf.float32,uniform=uniform)

def InitializerDeconvType(ksize,o_c,i_c):
    # i_c :input channel number
    # o_c:output channel number
    if i_c<o_c:
        raise ValueError('deconv filter weight error:outfilterNum is bigger than in channelNum')
    f = np.ceil(ksize/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([ksize, ksize],dtype = np.float32)
    for x in range(ksize):
        for y in range(ksize):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    vals = np.zeros([ksize,ksize,o_c,i_c])
    for j in range(o_c):
      for i in range(i_c):
        if np.mod(i-j,o_c)==0:
          vals[:, :, j, i] = bilinear
    return vals

def InitializerContextType(ksize,o_c,i_c):
    if i_c<o_c:
        raise ValueError('deconv filter weight error:outfilterNum is bigger than in channelNum')
    f = int(np.floor(ksize/2.0))
    std = 1.0/np.sqrt(ksize*ksize*o_c*i_c)
    limit = 1e-5
    vals = stats.truncnorm(-limit/std,limit/std, loc=0.0, scale=std).rvs([ksize,ksize,o_c,i_c]).astype(np.float32)
    for j in range(o_c):
      for i in range(i_c):
        if np.mod(i-j,o_c)==0:
          vals[f, f, j, i] = 1
    return vals

def Conv2DDilate(bottom,ksize,o_c,use_relu=True,atrous_rate = 1,name='',i_c=None,is_context_type=False):
  with tf.variable_scope(name):
    bshape = bottom.shape.as_list()
    if not i_c:
      kshape = [ksize,ksize]+[bshape[3]]+[o_c]
    else:
      kshape = [ksize,ksize]+[i_c]+[o_c]
    if is_context_type:
      vals = InitializerContextType(ksize,o_c,bshape[3])
      w = TensorWeights(shape = kshape,initializer=InitializerConstantType(vals))
    else:
      w = TensorWeights(shape = kshape,initializer=InitializerXavierType())
    b = TensorBias(shape = [o_c],initializer=InitializerConstantType(0))
    conv = tf.nn.atrous_conv2d(bottom, w, rate=atrous_rate, padding='SAME',name='atrous_conv2d')
    if use_relu:
      top = Relu(tf.nn.bias_add(conv,b,name='bias_add'))
    else:
      top = tf.nn.bias_add(conv,b,name='bias_add')
    print(bottom.name,'->',top.name)
    return top

#o_c output channel number
def Conv2D(bottom,ksize,stride,o_c,use_relu=True,name='',i_c=None):
  with tf.variable_scope(name):
    bshape = bottom.shape.as_list()
    if not i_c:
      kshape = [ksize,ksize]+[bshape[3]]+[o_c]
    else:
      kshape = [ksize,ksize]+[i_c]+[o_c]

    w = TensorWeights(shape = kshape,initializer=InitializerXavierType())
    b = TensorBias(shape = [o_c],initializer=InitializerConstantType(0))
    conv = tf.nn.conv2d(bottom, w, strides =[1,stride,stride,1], padding='SAME',name='conv2d')
    if use_relu:
      top = Relu(tf.nn.bias_add(conv,b,name='bias_add'))
    else:
      top = tf.nn.bias_add(conv,b,name='bias_add')
    print(bottom.name,'->',top.name)
    return top

def MaxPooling2D(bottom,ksize,stride,name=''):
  with tf.variable_scope(name):
    top = tf.nn.max_pool(bottom,ksize=[1, ksize, ksize, 1],strides=[1, stride, stride, 1],padding='SAME', name=name)
    print(bottom.name,'->',top.name)
    return top

def ConvPooling2D(bottom,ksize,stride,name=''):
  with tf.variable_scope(name):
    top = Conv2D(bottom,ksize,stride,bottom.shape.as_list()[-1],True,name='ConvPooling')
    return top

#top_sp: top tensor shape
def DeConv2D(bottom,ksize,stride,o_c,top_sp,usebias=False,name=''):
  with tf.variable_scope(name):
    f = np.ceil(ksize/2.0)
    if f!=stride:
      raise ValueError('ksize and stride are not compatible')
    bshape = bottom.shape.as_list()
    kshape = [ksize,ksize]+[o_c]+[bshape[3]]
    vals = InitializerDeconvType(ksize,bshape[3],o_c)
    w = TensorWeights(shape=kshape,initializer=InitializerConstantType(vals))
    strides = [1,stride,stride,1]
    deconv = tf.nn.conv2d_transpose(bottom, w, top_sp,strides=strides, padding='SAME',name='deconv')
    if usebias:
      b = TensorBias(shape=[o_c],initializer=InitializerConstantType(0))
      top = tf.nn.bias_add(deconv,b)
    else:
      top=deconv
    print(bottom.name,'->',top.name)
    return top

def Dropout(bottom,keep_prob=1.0,name=''):
  with tf.variable_scope(name):
    top = tf.nn.dropout(bottom,keep_prob=keep_prob,name=name)
    print(bottom.name,'->',top.name)
    return top

def Space2Batch(bottom,paddings,block_size=2,name=''):
  with tf.variable_scope(name):
    top = tf.space_to_batch(bottom,paddings = paddings,block_size = block_size,name=name)
    print(bottom.name,'->',top.name)
    return top

def Batch2Space(bottom,crops,block_size=2,name=''):
  with tf.variable_scope(name):
    top = tf.batch_to_space(bottom,crops = crops,block_size = block_size)
    print(bottom.name,'->',top.name)
    return top

def Relu(bottom,name='relu'):
  with tf.variable_scope(name):
    top = tf.nn.relu(bottom,name=name)
    return top

def Conv2DDilateMultiScale(bottom_list,ksize,o_c,use_relu=True,atrous_rate=2,name=''):
  top_list=[]
  top_list_tmp=[]
  with tf.variable_scope(name):
    print(name)
    top = None
    for idx in range(len(bottom_list)):
      with tf.variable_scope('scale_%i'%(idx)):
        bottom = bottom_list[idx]
        bshape = bottom.shape.as_list()
        kshape = [ksize,ksize]+[bshape[3]]+[o_c]
        w = TensorWeights(shape = kshape,initializer=InitializerXavierType())
        b = TensorBias(shape = [o_c],initializer=InitializerConstantType(0))
        conv = tf.nn.atrous_conv2d(bottom, w, rate=atrous_rate, padding='SAME',name='atrous_conv2d')
        conv_b = tf.nn.bias_add(conv,b,name='bias_add')
        top_list_tmp.append(conv_b)

    for idx in range(len(top_list_tmp)):
      with tf.variable_scope('scale_%i'%(idx)):
        top = top_list_tmp[idx]
        if use_relu:
          top = Relu(top)
        top_list.append(top)
    return top_list

#bottom_list in order of scale from smallest to biggest
def Conv2DMultiScale(bottom_list,ksize,stride,o_c,use_relu=True,name='',padding_list=None,concat=False,block_size=2,separate_towers=True):
  if padding_list is None:
    padding_list = [[[0,0],[0,0]]]*(len(bottom_list)-1)
  top_list=[]
  top_list_tmp=[]
  with tf.variable_scope(name):
    print(name)
    top = None
    for idx in range(len(bottom_list)):
      with tf.variable_scope('scale_%i'%(idx)):
        bottom = bottom_list[idx]
        bshape = bottom.shape.as_list()
        kshape = [ksize,ksize]+[bshape[3]]+[o_c]
        w = TensorWeights(shape = kshape,initializer=InitializerXavierType())
        b = TensorBias(shape = [o_c],initializer=InitializerConstantType(0))
        conv = tf.nn.conv2d(bottom, w, strides =[1,stride,stride,1], padding='SAME',name='conv2d')
        conv_b = tf.nn.bias_add(conv,b,name='bias_add')
        if not separate_towers:
          if idx!=0:
            top_upscale = Batch2Space(top,padding_list[idx-1],name='Batch2Space',block_size=block_size)
            if concat:
              conv_b = tf.concat([conv_b,top_upscale],axis=-1)
            else:
              conv_b = tf.add(conv_b,top_upscale)
        top = conv_b
        top_list_tmp.append(top)

    for idx in range(len(top_list_tmp)):
      with tf.variable_scope('scale_%i'%(idx)):
        top = top_list_tmp[idx]
        if use_relu:
          top = Relu(top)
        top_list.append(top)
    return top_list

#top_sp: top tensor shape
def DeConv2DMultiScale(bottom_list,ksize,stride,o_c,top_sp_list,usebias=False,name='',padding_list=None,concat=False,block_size=2,separate_towers=False):
  f = np.ceil(ksize/2.0)
  if f!=stride:
    raise ValueError('ksize and stride are not compatible')
  if padding_list is None:
    padding_list = [[[0,0],[0,0]]]*(len(bottom_list)-1)
  top_list=[]
  top_list=[]
  top_list_tmp=[]
  with tf.variable_scope(name):
    for idx in range(len(bottom_list)):
      with tf.variable_scope('scale_%i'%(idx)):
        top_sp = top_sp_list[idx]
        bottom = bottom_list[idx]
        bshape = bottom.shape.as_list()
        kshape = [ksize,ksize]+[o_c]+[bshape[3]]
        vals = InitializerDeconvType(ksize,o_c,bshape[3])
        w = TensorWeights(shape=kshape,initializer=InitializerConstantType(vals))
        strides = [1,stride,stride,1]
        deconv = tf.nn.conv2d_transpose(bottom, w, top_sp,strides=strides, padding='SAME',name='deconv')
        if usebias:
          b = TensorBias(shape=[o_c],initializer=InitializerConstantType(0))
          deconv = tf.nn.bias_add(deconv,b,name='bias_add')
        if not separate_towers:
          if idx!=0:
            top_upscale = Batch2Space(top,padding_list[idx-1],name='Batch2Space',block_size=block_size)
            if concat:
              deconv = tf.concat([deconv,top_upscale],axis=-1)
            else:
              deconv = tf.add(deconv,top_upscale)
        top = deconv
        top_list.append(top)
    return top_list

def ConvPooling2DMultiScale(bottom_list,ksize,stride,name=''):
  top_list = []
  with tf.variable_scope(name):
    for idx in range(len(bottom_list)):
      with tf.variable_scope('scale_%i'%(idx)):
        bottom = bottom_list[idx]
        top = ConvPooling2D(bottom,ksize,stride, name='ConvPooling')
        print(bottom.name,'->',top.name)
        top_list.append(top)
    return top_list

def MaxPooling2DMultiScale(bottom_list,ksize,stride,name=''):
  top_list = []
  with tf.variable_scope(name):
    for idx in range(len(bottom_list)):
      with tf.variable_scope('scale_%i'%(idx)):
        bottom = bottom_list[idx]
        top = tf.nn.max_pool(bottom,ksize=[1, ksize, ksize, 1],strides=[1, stride, stride, 1],padding='SAME', name=name)
        print(bottom.name,'->',top.name)
        top_list.append(top)
    return top_list

def DropoutMultiScale(bottom_list,keep_prob=1.0,name=''):
  top_list = []
  with tf.variable_scope(name):
    for idx in range(len(bottom_list)):
      with tf.variable_scope('scale_%i'%(idx)):
        bottom = bottom_list[idx]
        top = tf.nn.dropout(bottom,keep_prob=keep_prob,name=name)
        print(bottom.name,'->',top.name)
        top_list.append(top)
    return top_list

def AddMultiScale(bottom_list_1,bottom_list_2,name=""):
  top_list = []
  with tf.variable_scope(name):
    for idx in range(len(bottom_list_1)):
      with tf.variable_scope('scale_%i'%(idx)):
        bottom1 = bottom_list_1[idx]
        bottom2 = bottom_list_2[idx]
        top = tf.add(bottom1,bottom2)
        top_list.append(top)
    return top_list

def ShapeListForDeconv(bottom_list,i_c):
  top_sp = []
  for idx in range(len(bottom_list)):
    top_sp.append(tf.concat([tf.shape(bottom_list[idx])[:-1],[i_c]],axis=0))
  return top_sp

def TensorWeights(shape,initializer,wd=1e-5,dtype = tf.float32, name = 'weights',trainable = True):
  weight = tf.get_variable(name,shape=shape,initializer=initializer,dtype=dtype,trainable = trainable)
  tf.add_to_collection(L_weight_collection, weight)
  return weight

def TensorBias(shape,initializer,dtype = tf.float32, name = 'bias',trainable = True):
  bia = tf.get_variable(name,shape=shape,initializer=initializer,dtype=dtype,trainable = trainable)
  return bia

def SummaryVar(var):
  if not tf.get_variable_scope().reuse:
    name = var.op.name
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar(name + '/mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.summary.scalar(name + '/sttdev', stddev)
      tf.summary.scalar(name + '/max', tf.reduce_max(var))
      tf.summary.scalar(name + '/min', tf.reduce_min(var))
      tf.summary.histogram(name, var)
  return None