from IPython import display
import matplotlib.pylab as plt
import time
import numpy as np

def scatter(data):
    # data = [i*100 for i in data]
    plt.ylabel("Win %", color="C1")
    plt.scatter(range(len(data)), data)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    return True

def plot(data):
    data = [i*100 for i in data]
    plt.ylabel("Avg Reward over 100 games", color="0.0")
    plt.xlabel("n_eps / reward_window", color="0.0")
    plt.plot(data)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    return True

def plot_learning_curve(x, scores, epsilons):
    # Initialize double plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    # Set x-axis parameters
    ax1.set_xlabel("Episode #", color = "0.0")
    ax1.tick_params(axis="x", colors="0.0")
    
    # Set y-left parameters
    ax2.plot(x, scores, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel("Total Reward", color="C1")
    ax2.tick_params(axis="y", colors="C1")
    
    # Set y-right parameters
    ax1.plot(x, epsilons, color="C0")
    ax1.set_ylabel("Epsilon", color="C0")
    ax1.tick_params(axis="y", colors="C0")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    display.display(plt.gcf())
    display.clear_output(wait=True)
    return True
    
