class solverOptions:
  def __init__(self):
    #phase: 'Train' or 'Predict'
    self.phase='Train'
    self.test_interval=10000000
    self.base_lr=0.0001
    self.lr_policy='step'
    self.start_step=0
    self.stepsize=150
    self.gamma=0.96
    self.momentum=0.9
    self.weight_decay=0.005
    self.snapshot=20
    self.snapshow_prefix='./snapshot'
    self.solver_type='Adam'

    self.batch_size = 1
    self.epoch = 10
    self.pretrain_variables = None
    self.all_trainable_variables = None

  def __str__(self):
    info_str = ("Solver Configuration is as follows: \n"
    +"phase:%s \n"%(self.phase)
    +"test_interval:%d \n"%(self.test_interval)
    +"base_lr:%f \n"%(self.base_lr)
    +"lr_policy:%s \n"%(self.lr_policy)
    +"start_step:%d \n"%(self.start_step)
    +"stepsize:%d \n"%(self.stepsize)
    +"gamma:%f \n"%(self.gamma)
    +"momentum:%f \n"%(self.momentum)
    +"weight_decay:%f \n"%(self.weight_decay)
    +"snapshot:%d \n"%(self.snapshot)
    +"snapshow_prefix:%s \n"%(self.snapshow_prefix)
    +"solver_type:%s \n"%(self.solver_type))
    return info_str