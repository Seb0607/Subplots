#!/usr/bin/env python
# coding: utf-8

# In this app, we let the estimated GDP-CO2 relationship vary over time by including time as an additional input into the neural network. 

# In[67]:


get_ipython().run_line_magic('matplotlib', 'widget')

import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns; sns.set()

from ipywidgets import interact, fixed
import ipywidgets as widgets


# In[68]:


# Creating plotting function
def SurfPlot(region, loss, architecture):
    """
    Makes 3D surface plot for a given region in a given year.

    ARGUMENTS
        * region:       Must be from the list: 'World', 'OECD', 'Asia', 'REF', 'MAF', 'LAM'.
        * loss:         Must be from the list: 'MSE', 'MAE'
        * architecture: Must be from the list: (4), (8), (16), (8,4), (16,8,4) 
    """
    
    # Loading miscellaneous data
    time = np.load('Models/{}/Misc/time.npy'.format(region), allow_pickle=True)
    T    = np.load('Models/{}/Misc/T.npy'.format(region), allow_pickle=True)
    Min  = np.load('Models/{}/Misc/min.npy'.format(region), allow_pickle=True)
    Max  = np.load('Models/{}/Misc/max.npy'.format(region), allow_pickle=True)

    # Loading prediction model
    model_pred = tf.keras.models.load_model('Models/{}/{}/{}/model_pred'.format(region, loss, architecture))

    # PLotting
    ax1      = time - 1959
    ax2      = np.linspace(Min, Max, 1000)

    ax1, ax2 = np.meshgrid(ax1, ax2)

    ax1_vec  = np.reshape(ax1, (-1,1), order='F')
    ax2_vec  = np.reshape(ax2, (-1,1), order='F')
    
    x_input  = np.hstack((ax1_vec, ax2_vec))

    pred     = model_pred(x_input)
    pred     = np.reshape(np.array(pred), (1000, T), order='F')

    plt.close(None)
    fig  = plt.figure(figsize=[4.5,3])
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(ax1 + 1959, ax2, pred, linewidth=0, antialiased=True)
    ax.set_xlabel('year', fontweight='bold')
    ax.set_ylabel('log(GDP)', fontweight='bold')
    ax.set_zlabel('log(CO$_2$)', fontweight='bold')
    ax.set_zlim(-3,5)
    ax.view_init(36,-140)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    


# In[69]:


# Creating plotting function
def AggPlot(region, loss, architecture):
    """
    Makes aggregate predictions plot for a given region in a given year.

    ARGUMENTS
        * region:       Must be from the list: 'World', 'OECD', 'Asia', 'REF', 'MAF', 'LAM'.
        * loss:         Must be from the list: 'MSE', 'MAE'
        * architecture: Must be from the list: (4), (8), (16), (8,4), (16,8,4) 
    """
    
    # Loading miscellaneous data
    time        = np.load('Models/{}/Misc/time.npy'.format(region), allow_pickle=True)
    y_train_agg = np.load('Models/{}/Misc/y_train_agg.npy'.format(region), allow_pickle=True)

    # Loading specific data
    in_sample_agg = pd.read_pickle('Models/{}/{}/{}/in_sample_agg'.format(region, loss, architecture))

    # PLotting
    axis = np.reshape(np.array(time), (-1,1), order='F')
        
    plt.close(None)
    fig  = plt.figure(figsize=[4.5,3])
    if region=='REF':
        plt.plot(axis, in_sample_agg[5:], color='black', label='Model predictions')
        plt.plot(axis, y_train_agg[5:], color='black', linestyle='dashdot', label='Historical emissions')
    else:    
        plt.plot(axis, in_sample_agg, color='black', label='Model predictions')
        plt.plot(axis, y_train_agg, color='black', linestyle='dashdot', label='Historical emissions')
    plt.ylabel('Mt CO$_2$', fontweight='bold')
    plt.xlabel('year', fontweight='bold')
    if region=='World':
        plt.ylim(2000,45000)
    elif region=='OECD':
        plt.ylim(2000,15000)
    elif region=='REF':
        plt.ylim(0,4500)
    elif region=='Asia':
        plt.ylim(0,25000)
    elif region=='MAF':
        plt.ylim(0,4000)
    elif region=='LAM':
        plt.ylim(0,2000)    
    plt.legend(loc='upper left', fancybox=True, shadow=True, fontsize='x-small')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


# ### Estimated Surface

# In[70]:


interact(SurfPlot, region=['World', 'OECD', 'REF', 'Asia', 'MAF', 'LAM'], loss=['MSE', 'MAE'], architecture=['(4)', '(8)', '(16)', '(8,4)', '(16,8,4)']);


# ### Aggregate In-sample Predictions

# In[71]:


interact(AggPlot, region=['World', 'OECD', 'REF', 'Asia', 'MAF', 'LAM'], loss=['MSE', 'MAE'], architecture=['(4)', '(8)', '(16)', '(8,4)', '(16,8,4)']);


# In[38]:


#plt.close('all')

