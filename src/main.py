import os
import tensorflow as tf

from ini_file_io import load_train_ini
from model import unet_2D_xy

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main(_):
    # load training parameter #
    ini_file = '../outcome/model/ini/tr_param.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]

    print '====== Phase >>> %s <<< ======' % param_set['phase']

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])

    # GPU setting, per_process_gpu_memory_fraction means 95% GPU MEM ,allow_growth means unfixed memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = unet_2D_xy(sess, param_set)
        if param_set['phase'] == 'train':
            # train the network in a self-play way
            model.train_online_seg()
        elif param_set['phase'] == 'test':
            # test in an online way
            model.test_online_seg()

if __name__ == '__main__':
    tf.app.run()
