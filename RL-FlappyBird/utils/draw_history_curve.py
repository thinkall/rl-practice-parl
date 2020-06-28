import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def draw_results(train_score_list, test_score_list, title='', path='./fig_dir/'):
    # 画出训练过程reward历史曲线
    fig = plt.figure(figsize=[10, 4])
    plt.subplot(121)
    plt.plot(train_score_list, color='green', label='train')
    plt.title('Train History {}'.format(title))
    plt.legend()
    plt.subplot(122)
    x = list(range(len(test_score_list)))
    plt.plot([i*50 for i in x],test_score_list, color='red', label='test')
    plt.title('Test History {}'.format(title))
    plt.legend()
    if path != '' and not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + title + '.png')
    plt.show()


with open('../train.pickle', 'rb') as f:
    data = pickle.load(f)
    draw_results(data[0], data[1], title='FlappyBird')
