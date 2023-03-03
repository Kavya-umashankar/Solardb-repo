import sys
import os

flares = ['C', 'M', 'M5']

start_feature = 5
mask_value = 0
series_len = 10
epochs = 7
batch_size = 256
nclass = 2


def get_flare_category(args):
    if not 'flare' in args.keys():
        args['flare'] = 'C'
    flare_label = args['flare']
    if not flare_label.strip().upper() in flares:
        print('Invalid flare category:', flare_label, '\nFlare category must one of: ', flares)
        sys.exit()
    flare_label = flare_label.strip().upper()
    args['flare'] = flare_label
    return flare_label


def get_training_input(args):
    TRAIN_INPUT = args['train_data_file']
    if TRAIN_INPUT.strip() == '':
        print('Training data file can not be empty')
        sys.exit()
    if not os.path.exists(TRAIN_INPUT):
        print('Training data file does not exist:', TRAIN_INPUT)
        sys.exit()
    if not os.path.isfile(TRAIN_INPUT):
        print('Training data is not a file:', TRAIN_INPUT)
        sys.exit()
    return TRAIN_INPUT


def get_test_input(args):
    TEST_INPUT = args['test_data_file']
    if TEST_INPUT.strip() == '':
        print('Testing data file can not be empty')
        sys.exit()
    if not os.path.exists(TEST_INPUT):
        print('Testing data file does not exist:', TEST_INPUT)
        sys.exit()
    if not os.path.isfile(TEST_INPUT):
        print('Testing data is not a file:', TEST_INPUT)
        sys.exit()
    return TEST_INPUT


def get_model_id(args):
    modelid = args['modelid']
    if modelid.strip() == '':
        print('Model id can not be empty')
        sys.exit()
    return modelid


def get_models_dir(flare_label):
    return './pretrained_model/LSTM_{flare_label}_sample_run'.format(flare_label=flare_label)


def get_model_dir(flare_label, model_id):
    models_dir = get_models_dir(flare_label)
    return models_dir + '/{model_id}.h5'.format(flare_label=flare_label,
                                                model_id=model_id)

def get_n_features(flare_label):
    if flare_label == 'C':
        n_features = 14
    elif flare_label == 'M':
        n_features = 22
    elif flare_label == 'M5':
        n_features = 20
    return n_features

def are_model_files_exist(models_dir, modelId, flare='C'):
    flare = str(flare).strip().upper()
    # log("Searching for model is: " + modelId + " in directory: " + models_dir)
    modelExenstion = ".h5"
    fname = models_dir + "/" + modelId + modelExenstion
    return os.path.isfile(fname)

def get_result_file(flare_label, model_id):
    return './results/LSTM_{flare_label}_sample_run/{model_id}.csv'.format(flare_label=flare_label, model_id = model_id)