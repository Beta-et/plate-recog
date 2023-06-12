import os
import glob
import shutil
import subprocess

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
region_threshold = 0.4
detection_threshold = 0.7

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc'),
    'RESEARCH':os.path.join('Tensorflow', 'models', 'research'),
    'SAVE_FILE':os.path.join('detection_images')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

file_indicator = False
for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            subprocess.call(['mkdir', '-p', path])

for name, path in paths.items():
    if os.path.exists(path):
        file_indicator = True
        if name == 'RESEARCH':
            shutil.rmtree(paths['APIMODEL_PATH'])
            print('APIMODEL_PATH removed')
    else:
        file_indicator = False
        break

def prepper():
    print('All files found. Proceeding to next step...')
    subprocess.call(['cp', '-r', 'test', paths['IMAGE_PATH']])
    subprocess.call(['cp', '-r', 'train', paths['IMAGE_PATH']])
    # Install Tensorflow Object Detection
    # if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    subprocess.call(['git', 'clone', 'https://github.com/tensorflow/models', paths['APIMODEL_PATH']])

    if os.name=='posix':  
        # subprocess.call(['apt-get', 'install', 'protobuf-compiler'])
        # subprocess.call(['protoc', 'object_detection/protos/*.proto', '--python_out=.', '&&', 'cp', 'object_detection/packages/tf2/setup.py', '.', '&&', 'python', '-m', 'pip', 'install', '.']) 
        # current_dir = os.chdir(paths['RESEARCH'])
        current_dir = os.path.dirname(os.path.abspath('Tensorflow/models/research/'))
        protos_path = os.path.join(current_dir, 'research/object_detection', 'protos')
        print(protos_path)
        proto_files = glob.glob(os.path.join(protos_path, '*.proto'))
        protoc_command = ['protoc', '--proto_path', protos_path, '--python_out=.']
        protoc_command.extend(proto_files)
        subprocess.call(protoc_command)
        subprocess.call(['cp', 'object_detection/packages/tf2/setup.py', '.'])
        subprocess.call(['pip', 'install', '.'])    

    # Verify Installation
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    subprocess.call(['python', VERIFICATION_SCRIPT])

    import object_detection
    # Dowload TensorFlow Pretrained Models from the Zoo
    subprocess.call(['wget', PRETRAINED_MODEL_URL])
    subprocess.call(['mv', PRETRAINED_MODEL_NAME+'.tar.gz', paths['PRETRAINED_MODEL_PATH']])
    os.chdir(paths['PRETRAINED_MODEL_PATH'])
    subprocess.call(['tar', '-zxvf', PRETRAINED_MODEL_NAME+'.tar.gz'])

    # create label map
    labels = [{'name':'licence', 'id':1}]
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        subprocess.call(['git', 'clone', 'https://github.com/nicknochnack/GenerateTFRecord', paths['SCRIPTS_PATH']])

    subprocess.call(['python', files['TF_RECORD_SCRIPT'], '-x', os.path.join(paths['IMAGE_PATH'], 'train'), '-l', files['LABELMAP'], '-o', os.path.join(paths['ANNOTATION_PATH'], 'train.record')])
    subprocess.call(['python', files['TF_RECORD_SCRIPT'], '-x', os.path.join(paths['IMAGE_PATH'], 'test'), '-l', files['LABELMAP'], '-o', os.path.join(paths['ANNOTATION_PATH'], 'test.record')])

    subprocess.call(['cd', os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'), os.path.join(paths['CHECKPOINT_PATH'])])

    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)
