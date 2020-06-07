import tensorflow as tf
import numpy as np
import LLayer as L
import os

def setUpSummary(graph,summaryDir,summaryName):
  assert(graph is not None)
  summaryPath = os.path.join(summaryDir,summaryName)
  writer = tf.summary.FileWriter(summaryPath)
  writer.add_graph(graph)
  
def addSummaryWeights(weights):
  for w in weights:
    with tf.name_scope('summray'):
      L.SummaryVar(w)

def addSummaryLoss(loss,name):
  tf.summary.scalar(name,loss)

def addSummaryAccuracy(accuracy,name):
  tf.summary.scalar(name,accuracy)