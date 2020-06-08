import PIL
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue
import copy

class LOutOfRangeError(Exception):
  pass

class DataGenerator(object):
  def __init__(self,fileList,output_channels,batch_size,epoch,shuffle=True,threadBuf=3,threadNum = 1):
    print("Build Data Generator..")
    self.fileList = fileList
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.temp_fileList = copy.copy(fileList)
    self.output_channels = output_channels
    self.label_encoder = None
    assert(epoch>0)
    self.epoch = epoch
    self.data_queue = Queue(threadBuf)
    self.threadNum = threadNum
    self.FinishGen = mp.Value('b',False)
    self.p = None

  def __del__(self):
    if self.p:
      for it in self.p:
        it.terminate()

  def enQueueThread(self,threadNum,threadId):
    count=0
    while count<self.epoch:
      count+=1
      fileList = self.fileList
      if self.shuffle:
        self.temp_fileList = copy.copy(fileList)
        np.random.shuffle(self.temp_fileList)

      num = len( self.temp_fileList)
      imax = int(num//self.batch_size)
      for i in range(threadId,imax,threadNum):
        batch_list = [it for it in  self.temp_fileList[i*self.batch_size:(i+1)*self.batch_size]]
        X,y = self.__data_generation(batch_list)
        self.data_queue.put((X,y))
    self.FinishGen.value = True

  def generate(self):
    if self.p is None:
      self.p = []
      for idx in range(self.threadNum):
        self.p.append(mp.Process(target = self.enQueueThread,args=(self.threadNum,idx)))
      for it in self.p:
        it.start()

      assert(self.p is not None)
      print('EnqueueThread started.')
    while True:
      if self.FinishGen.value==True and self.data_queue.empty():
        print('Finish input.')
        print('Terminate thread for reader')
        for it in self.p:
          it.terminate()
        raise LOutOfRangeError()
        break
      else:
        X,y = self.data_queue.get()
        yield X,y
    
  def generateImg(self):
    for path in self.fileList:
      X = self.__img_data_generation([path])
      yield X
    print('Generating Images Finished.')

  def __data_generation(self,batch_list):
    X=[]
    y=[]
    for it in batch_list:
      paths = it.split()
      imgP = paths[0]
      lblP = paths[1]
      tx = np.array(PIL.Image.open(imgP))
      ty = np.array(PIL.Image.open(lblP))
      tx,ty = self.__preprocess(tx,ty,True)
      X.append(tx)
      y.append(ty)
    X = np.array(X, dtype = np.float32)
    y = np.array(y,dtype = np.int32)
    return X,y

  def __img_data_generation(self,batch_list):
    X=[]
    for it in batch_list:
      imgP = it
      tx = np.array(PIL.Image.open(imgP))
      tx,_ = self.__preprocess(tx,None,random_flip_lr = False)
      X.append(tx)
    X = np.array(X)
    return X

  def __one_hot(self,label):
    o_c = self.output_channels
    outLabel = np.zeros(shape=(label.shape[0],label.shape[1],o_c))
    for i in range(o_c):
      mask = (label==i)
      outLabel[:,:,i][mask] = 1
    assert(outLabel.shape[-1]==o_c)
    return outLabel

  def set_label_encoder(self,encoder):
    self.label_encoder = encoder

  def __encode_label(self,label):
    return self.label_encoder.transform(label)

  def __preprocess(self,img,lbl,is_int_label = False,random_flip_lr = True):
    if img is not None:
      img = img.astype(np.float32)/255.0
    if lbl is not None and not is_int_label:
      lbl = self.__encode_label(lbl)
    if lbl is not None and random_flip_lr:
      if np.random.choice([0,1],p=[0.5,0.5])==1:
        img = np.fliplr(img)
        lbl = np.fliplr(lbl)
    return img,lbl

def getPairFileLists(listFilePath):
  fileList=[]
  with open(listFilePath,'r') as f:
    print(listFilePath+'--file open success')
    lines = f.read().split()
  for idx in range(0,len(lines)//2):
    fileList.append(lines[idx*2]+' '+lines[idx*2+1])
  return fileList

#Text file whose each row formed by <image> only
def getSingleFileLists(listFilePath):
  with open(listFilePath,'r') as f:
    print(listFilePath+'--file open success')
    lines = f.read().split()
  return lines

#Text file whose each row formed by <image,groundTruth,prediction>
def getTripleFileLists(listFilePath):
  imageFileList = []
  groundTruthFileList = []
  predictionFileList = []
  with open(listFilePath,'r') as f:
    print(listFilePath+'--file open success')
    lines = f.read().split('\n')
  for line in lines:
    items = line.split()
    if len(items)==3:
      imageFileList.append(items[0])
      groundTruthFileList.append(items[1])
      predictionFileList.append(items[2])
  print('Prediction file number is: %d'%(len(predictionFileList)))
  return imageFileList,groundTruthFileList,predictionFileList

#Text file whose each row formed by <image,prediction>
def getDoubleFileLists(listFilePath):
  imageFileList = []
  predictionFileList = []
  with open(listFilePath,'r') as f:
    print(listFilePath+'--file open success')
    lines = f.read().split('\n')
  for line in lines:
    items = line.split()
    if len(items)==2:
      imageFileList.append(items[0])
      predictionFileList.append(items[1])
  print('Prediction file number is: %d'%(len(predictionFileList)))
  return imageFileList,predictionFileList
