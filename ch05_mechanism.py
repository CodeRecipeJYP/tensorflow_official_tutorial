from toolbox import *
from graphtoolbox import *
from runtoolbox import *

class Hyparms():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.1
        self.max_steps = 20000
        self.log_dir = 'logs'
        self.input_data_dir = 'MNIST_data'
        self.dropout_rate = 1
        self.fake_data = False

def load_params():
    HYPARMS = Hyparms()

    # Set parameters
    #HYPARMS.max_steps = 200
    #HYPARMS.learning_rate = 0.1
    #HYPARMS.batch_size = 128
    #HYPARMS.log_dir = "logs"
    #HYPARMS.dropout_rate = 0.8
    #HYPARMS.input_data_dir = 'MNIST_data'

    return HYPARMS

def main(_):
    HYPARMS = load_params()
    if tf.gfile.Exists(HYPARMS.log_dir):
        tf.gfile.DeleteRecursively(HYPARMS.log_dir)
    tf.gfile.MakeDirs(HYPARMS.log_dir)
    run_training(HYPARMS)

if __name__ == '__main__':
  tf.app.run(main=main)