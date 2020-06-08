import tensorflow as tf
import LLayer

def tf_multiscale_sparse_softmax_cross_entropy_with_logits(y_logits_list,y_true_list,loss_weights=[0.7,1.0,1.3],ignore_label=255,loss_weight=1.0):
  loss_list=[]
  for idx in range(len(y_logits_list)):
    loss = tf_sparse_softmax_cross_entropy_with_logits(y_logits_list[idx],y_true_list[idx],ignore_label,loss_weight)*loss_weights[idx]/3.0
    loss_list.append(loss)
  total_loss = tf.add_n(loss_list)
  return total_loss

#y_true should not be one hot encoded
def tf_sparse_softmax_cross_entropy_with_logits(y_logits,y_true,ignore_label,loss_weight=1.0):
  not_ignore_mask = tf.to_float(tf.not_equal(y_true,ignore_label)) * loss_weight
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_logits)*not_ignore_mask
  loss =tf.reduce_mean(cross_entropy)
  return loss

def multiscale_sparse_softmax_cross_entropy_with_logits(y_logits_list,y_true_list,class_num,loss_weights=[1,1,1],ignore_label=255,loss_weight=1.0):
  loss_list=[]
  for idx in range(len(y_logits_list)):
    loss = sparse_softmax_cross_entropy_with_logits(y_logits_list[idx],y_true_list[idx],class_num,ignore_label,loss_weight)*loss_weights[idx]/3.0
    loss_list.append(loss)
  total_loss = tf.add_n(loss_list)
  return total_loss


def sparse_softmax_cross_entropy_with_logits(y_logits,y_true,class_num,ignore_label,loss_weight):
  with tf.variable_scope('loss_sparse_softmax_cross_entropy_with_logits'):
    not_ignore_mask = tf.to_float(tf.not_equal(y_true,ignore_label)) * loss_weight
    y_one_hot = tf.one_hot(y_true,depth=class_num,axis=-1,dtype=tf.float32)
    q = tf.nn.softmax(y_logits,dim=-1)
    logsoftmax = -tf.log(tf.add(q,1e-8))
    cross_entropy = tf.reduce_mean(tf.multiply(y_one_hot,logsoftmax),axis=-1)
    loss = tf.reduce_sum(cross_entropy*not_ignore_mask)/(tf.reduce_sum(not_ignore_mask)+1e-8)
    return loss

def regularization_loss(wd):
  if wd==0:
      return
  else:
    print('adding weight decay...')
    weight_collection = tf.get_collection(LLayer.L_weight_collection)
    print("weight_collection length:",len(weight_collection))
  for w in weight_collection:
        AddWeightDecay(w,wd)

  regLossList = tf.get_collection(LLayer.L_weight_decay_loss_collection)
  if not isinstance(regLossList,list):
    raise TypeError('RegularizationLoss function should use a list of regloss as arguments')
  loss = tf.add_n(regLossList)
  return loss

def AddWeightDecay(var,wd=1e-5):
  with tf.variable_scope('weight_decay'):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
    tf.add_to_collection(LLayer.L_weight_decay_loss_collection, weight_decay)
    return weight_decay

