import matplotlib.pyplot as plt
import random

def init_plt():
    plt.title("loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend(["train_loss"])
    plt.ion()
    plt.show()

def draw(iters, losses):
    train_loss_lines = plt.plot(iters, losses,'r',lw=1)
    plt.draw()
    plt.pause(1)

if __name__ == '__main__':

    
    iters = []
    losses = []
    for i in range(20):
        iters.append(i)
        losses.append(random.randint(0, 100))
        train_loss_lines = plt.plot(iters, losses,'r',lw=1)
        plt.draw()
        plt.pause(1)
        # plt.pause(0.1)

