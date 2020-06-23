# import necessary packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata


def train_loss_vis(train_loss_list, val_loss_list):
    plt.figure()
    plt.plot(train_loss_list, label='training loss')
    plt.plot(val_loss_list, label='validation loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def pred_test_vis(measured_data, pred_data):
    plt.figure()
    plt.scatter(measured_data, pred_data, marker='o')
    plt.plot([0, 1], [0, 1], color='r')
    plt.xlabel('Measured data')
    plt.ylabel('Predicted data')
    plt.show()
