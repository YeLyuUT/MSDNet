import tensorflow as tf
from LDataInput import *

class TFGenerator:
  def __init__(self,sess,threadNum,capacity,dtypes,shapes=None,names=None,shared_name=None,name = 'fifo_queue'):
    self.sess = sess
    self.threadNum = threadNum
    self.capacity = capacity
    self.dtypes = dtypes
    self.shapes = shapes,
    self.names = names,
    self.shared_name = shared_name,
    self.name = name
    self.coord = None
    self.Input_X = tf.placeholder(dtype=dtypes,shape = shapes)
    self.enqueue_op = None
    self.queue = None
    self.gen = None
    assert(threadNum>=1)
    assert(capacity>=1)

  def create_queue(self,data_gen):
    print('creating FIFO queue')
    self.queue = tf.FIFOQueue(self.capacity,self.dtypes,self.shapes)
    self.enqueue_op = self.queue.enqueue(self.Input_X)
    self.gen = data_gen
    assert(self.enqueue_op is not None)
    assert(self.queue is not None)
    assert(self.gen is not None)
        
  def start_enqueue_loop(self):
    print('start enqueue loop')
    qr = tf.train.QueueRunner(self.queue,[self.enqueue_op]*self.threadNum)
    if self.coord is None:
      self.coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(self.sess,coord = self.coord,start = True)
    for X,y in self.gen():
      if self.coord.should_stop():
        break
      else:
        print(X.shape)
        self.sess.run(self.enqueue,feed_dict = {self.Input_X:X})

    self.coord.request_stop()
    self.coord.join(enqueue_threads)

  def dequeue(self):
    X,y = self.queue.dequeue()
    return X,y

