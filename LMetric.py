import tensorflow as tf

#y_true should not be one hot encoded
def accuracy(y_pred,y_true,ignore_label=255):
  with tf.name_scope('accuracy'):
    not_ignore_mask = tf.to_float(tf.not_equal(y_true,ignore_label))
    accuracy = tf.reduce_sum(tf.cast(tf.equal(y_pred,y_true) ,tf.float32)*not_ignore_mask)/(tf.reduce_sum(not_ignore_mask)+1e-8)
    return accuracy

def accuracy_multiscale(y_pred_list,y_true_list,ignore_label=255):
  print('y_pred_list:',len(y_pred_list))
  print('y_true_list:',len(y_true_list))
  accuracy=[]
  with tf.name_scope('multiscale_accuracy'):
    for idx in range(len(y_pred_list)):
      not_ignore_mask = tf.to_float(tf.not_equal(y_true_list[idx],ignore_label))
      acc = tf.reduce_sum(tf.cast(tf.equal(y_pred_list[idx],y_true_list[idx]) ,tf.float32)*not_ignore_mask)/(tf.reduce_sum(not_ignore_mask)+1e-8)
      accuracy.append(acc)
    return accuracy


def accuracy_for_all(y_pred,y_true,ClassNum):
  #first classNum accuracies are for each class
  #last accuracy is for total accuracy
  #TODO fix this..
  accuracies = []
  print('y_true',y_true.get_shape().as_list())
  with tf.name_scope('accuracy_for_all'):
    for idx in range(ClassNum):
      mask = tf.not_equal(y_true,tf.constant(idx))
      print('mask',mask.get_shape().as_list())
      s_y_pred = tf.boolean_mask(y_pred,mask)
      print('s_y_pred',s_y_pred.get_shape().as_list())
      s_y_true = tf.boolean_mask(y_true,mask)
      print('s_y_true',s_y_true.get_shape().as_list())
      correct_labels = tf.equal(s_y_pred,s_y_true) 
      print('correct_labels',correct_labels.get_shape().as_list())
      accuracy = tf.reduce_mean(tf.cast(correct_labels,tf.float32))
      print('accuracy',accuracy.get_shape().as_list())
      accuracies.append(accuracy)

    accuracy = tf.reduce_mean(tf.concat(accuracies,axis = 0))
    accuracies.append(accuracy)
    all_accuracy = tf.concat(accuracies,axis = 0)
    return all_accuracy