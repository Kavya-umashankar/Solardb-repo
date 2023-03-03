# =========================================================================
#   (c) Copyright 2019
#   All rights reserved
#   Programs written by Hao Liu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.models import *
from keras.layers import *
import csv
import numpy as np
import os
import warnings
from tensorflow.python.keras import regularizers

warnings.filterwarnings("ignore")

class LSTM_Flare:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        try:
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except Exception as e:
            print('turn off loggins is not supported')

    def check_label(self, flare_label, flare):
        label = flare[0]
        if flare_label == 'C':
            if label == 'X' or label == 'M':
                label = 'C'
            elif label == 'B':
                label = 'N'
        elif flare_label == 'M':
            if label == 'X':
                label = 'M'
            elif label == 'B' or label == 'C':
                label = 'N'
        elif flare_label == 'M5':
            if label == 'M':
                scale = flare[1:]
                if float(scale) >= 5.0:
                    label = 'X'
                else:
                    label = 'N'
            elif label == 'C' or label == 'B':
                label = 'N'
        return label

    def check_zero_record(self, flare_label, row):
        if flare_label == 'C':
            cols = [5, 7] + list(range(9, 13)) + list(range(14, 16)) + [18]
        elif flare_label == 'M':
            cols = list(range(5, 10)) + list(range(13, 16)) + [19, 21] + list(range(23, 26))
        elif flare_label == 'M5':
            cols = list(range(5, 12)) + list(range(19, 21)) + list(range(22, 25))
        for k in cols:
            if float(row[k]) == 0.0:
                return True
        return False

    def load_data(self, datafile, flare_label, series_len, start_feature, n_features, mask_value):
        df = pd.read_csv(datafile)
        df_values = df.values
        X = []
        y = []
        tmp = []
        for k in range(start_feature, start_feature + n_features):
            tmp.append(mask_value)
        for idx in range(0, len(df_values)):
            each_series_data = []
            row = df_values[idx]
            flare = row[1]
            label = self.check_label(flare_label, flare)
            has_zero_record = False
            # if at least one of the 25 physical feature values is missing, then discard it.
            has_zero_record = self.check_zero_record(flare_label, row)

            if has_zero_record is False:
                cur_noaa_num = int(row[3])
                each_series_data.append(row[start_feature:start_feature + n_features].tolist())
                itr_idx = idx - 1
                while itr_idx >= 0 and len(each_series_data) < series_len:
                    prev_row = df_values[itr_idx]
                    prev_noaa_num = int(prev_row[3])
                    if prev_noaa_num != cur_noaa_num:
                        break
                    has_zero_record_tmp = self.check_zero_record(flare_label, row)

                    if len(each_series_data) < series_len and has_zero_record_tmp is True:
                        each_series_data.insert(0, tmp)

                    if len(each_series_data) < series_len and has_zero_record_tmp is False:
                        each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                    itr_idx -= 1

                while len(each_series_data) > 0 and len(each_series_data) < series_len:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) > 0:
                    X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                    y.append(label)
        X_arr = np.array(X)
        y_arr = np.array(y)
        print(X_arr.shape)
        return X_arr, y_arr

    def data_transform(self, data):
        encoder = LabelEncoder()
        encoder.fit(data)
        encoded_Y = encoder.transform(data)
        converteddata = np_utils.to_categorical(encoded_Y)
        return converteddata

    def attention_3d_block(self, hidden_states, series_len):
        hidden_size = int(hidden_states.shape[2])
        hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
        hidden_states_t = Reshape((hidden_size, series_len), name='attention_input_reshape')(hidden_states_t)
        score_first_part = Dense(series_len, use_bias=False, name='attention_score_vec')(hidden_states_t)
        score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
        h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
        score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
        context_vector = Reshape((hidden_size,))(context_vector)
        h_t = Reshape((hidden_size,))(h_t)
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(
            pre_activation)
        return attention_vector

    def lstm(self, nclass, n_features, series_len):
        inputs = Input(shape=(series_len, n_features,))
        lstm_out = LSTM(10, return_sequences=True, dropout=0.5)(inputs)
        attention_mul = self.attention_3d_block(lstm_out, series_len)
        layer1_out = Dense(200, activation='relu')(attention_mul)
        layer2_out = Dense(500, activation='relu')(layer1_out)
        output = Dense(nclass, activation='softmax', activity_regularizer=regularizers.l2(0.0001))(layer2_out)
        model = Model(inputs, output)
        return model
