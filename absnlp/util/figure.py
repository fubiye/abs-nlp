import matplotlib.pyplot as plt
import random
global loss_plt, f1_plt
loss_color = 'tab:red'
f1_color = 'tab:green'

def init_plt():
    global loss_plt, f1_plt
    plt.title("loss")
    fig, loss_plt = plt.subplots()
    f1_plt = loss_plt.twinx()

    loss_plt.set_xlabel("iterations")
    loss_plt.set_ylabel("loss", color=loss_color)
    loss_plt.tick_params(axis='y', labelcolor=loss_color)

    f1_plt.set_ylabel("f1", color=f1_color)
    f1_plt.tick_params(axis='y', labelcolor=f1_color)
    fig.tight_layout()
    plt.ion()
    plt.show()
def draw(iters, losses, f1_scores):
    global loss_plt, f1_plt
    loss_plt.plot(iters, losses,color=loss_color,lw=1)
    f1_plt.plot(iters, f1_scores,color=f1_color, lw=1)
    plt.draw()
    plt.pause(1)

if __name__ == '__main__':

    init_plt()
    iters = []
    losses = []
    f1_scores = []
    for i in range(20):
        iters.append(i)
        losses.append(random.randint(0, 100))
        f1_scores.append(random.randint(0, 100))
        draw(iters, losses, f1_scores)
        # train_loss_lines = plt.plot(iters, losses,'r',lw=1)
        # plt.draw()
        # plt.pause(1)
        # plt.pause(0.1)

