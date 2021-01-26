from matplotlib import pyplot as plt


def plot_training(history, x_lim=None, y_lim=None):
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(history.history["loss"], color="red", alpha=.4, label="Training")
    ax.plot(history.history["val_loss"], color="black", alpha=.4, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()


def plot_learning_rate(history, x_lim=None, y_lim=None, text=None):
    """Plot loss vs learning rate. Only works if lr scheduler has been invoked"""
    fig, ax = plt.subplots(figsize=(15, 3))
    label = "Training"
    if text:
        label = label + " " + text
    ax.semilogx(history.history["lr"], history.history["loss"], color="black", label=label)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.grid()
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
