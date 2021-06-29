# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:07:21 2021

@author: Oleg Kryachun
"""

# Pie Plot
import matplotlib.pyplot as plt
import os

def pie_plot(column, title, save=False):
    """Take column from DataFrame and create pie graph

    Parameters:
    ----------
        column (pd.Dataframe): column
        title (str): pie graph title
        save (bool): determines if plot images should be saved to disk
    """
    filename = os.path.join("graphs", "data_stats", title)

    labels = column.value_counts().index.tolist()
    counts = column.value_counts().values.tolist()

    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=labels, autopct='%1.2f%%',
            radius=1.5, startangle=90)
    ax1.set_title(title, fontdict={'fontsize':20}, y=1.2)

    if save:
        plt.savefig(filename)

    plt.show()
    


def plot_model_measure(n, train_var, valid_var, plot_title, y_title, save= False):
    """Plot scatter plot for model training statistics

    Parameters
    ----------
        n (int): number for lot frame
        train_var (list): model history train attribute
        valid_var (list): model history validation attribute 
        plot_title (str): main plot title
        y_title (str): y-axis title
        save (bool): determines if plot images should be saved to disk
    """
    filename = os.path.join("graphs", "model_stats", plot_title)
    plt.figure(n)
    plt.plot(train_var)
    plt.plot(valid_var)
    plt.title(plot_title)
    plt.ylabel(y_title)
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    
    if save:
        plt.savefig(filename)

    plt.show()

    






