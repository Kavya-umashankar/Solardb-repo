'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
    
import numpy as np 
import pandas as pd 
from datetime import datetime

from KpNet_utils import *
from KpNet_model import KpNetModel

interval_type = 'hourly'
show_figures=True
figures_dir = 'figures'
results_dir='results'

def plot_figures(start_hour, end_hour,show_figures=False,figures_dir = figures_dir,results_dir=results_dir):
    os.makedirs(results_dir,  exist_ok=True)
    os.makedirs(figures_dir,  exist_ok=True)    
    for k in range(start_hour,end_hour):
        print('Graphing results for h =', k, 'hour ahead')
        num_hours = k
        model = KpNetModel()
        predictions,ale,epis,y_test,ax_dates = model.get_results(num_hours,results_dir=results_dir)
        
        ax_dates = [datetime.strptime(t,'%Y-%m-%d %H:%M:%S') for t in ax_dates]
        
        file_name = figures_dir +'' + os.sep + 'kp_' +str(num_hours) + interval_type[0] +'.png'
        file_name_aleatoric = figures_dir +'' + os.sep + 'kp_' +str(num_hours) + interval_type[0] +'_aleatoric.png'
        file_name_epistemic = figures_dir +'' + os.sep + 'kp_' +str(num_hours) + interval_type[0] +'_epistemic.png'
        
        figsize = (5,2)
        plot_figure(ax_dates,y_test,predictions,epis,num_hours,label='Kp index',
                    file_name=file_name_epistemic ,wider_size=False,
                    interval=interval_type[0],uncertainty_label='Epistemic Uncertainty', 
                    fill_graph=True,
                    prediction_color='black',
                    observation_color='gold',                     
                    x_labels=True,
                    x_label='Time\n(a)')
        plot_figure(ax_dates,y_test,predictions,ale,num_hours,label='Kp index',
                    file_name=file_name_aleatoric ,wider_size=False,
                    interval=interval_type[0],uncertainty_label='Aleatoric Uncertainty', fill_graph=True,
                    uncertainty_color='#aabbff',
                    prediction_color='black',
                    observation_color='gold',                     
                    x_labels=True,
                    x_label='Time\n(b)')        

    log('show_figures:', show_figures)
    if show_figures:
        plt.show()

if __name__ == '__main__':
    starting_hour = 1
    ending_hour = 10
    if len(sys.argv) >= 2 :
        if not sys.argv[1].lower() == 'all':
            starting_hour = int(float(sys.argv[1]))
            ending_hour = starting_hour + 1
        if sys.argv[1].lower() == 'all':
            starting_hour = 1
            ending_hour = 10
                    
    if len(sys.argv) >= 3:
        show_figures = boolean(sys.argv[2])
    
    if starting_hour < 1 or starting_hour-1 > 9:
        print('Invalid starting hour:', starting_hour,'\nHours must be between 1 and 9.')
        exit() 
    print('Starting hour:', starting_hour, 'ending hour:', ending_hour-1)
    plot_figures(starting_hour, ending_hour, show_figures)