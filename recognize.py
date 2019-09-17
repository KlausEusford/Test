import image
import network_img as nw
import random
import numpy as np
#取最大值
def max_index(arry):
    max = -10
    max_index = 0
    for i in range(len(arry)):
        if arry[i] > max:
            max = arry[i]
            max_index = i
    return max_index

def error_eva(net):
    accuracy = 0
    for i in range(100):
        word_i = random.randint(1, 14)
        image_i = random.randint(240, 255)
        file_name = './TEST/' + str(word_i) + '/' + str(image_i) + '.bmp'
        inputs = np.array(image.pic(file_name))
        outputs = net.compute(inputs)
        if max_index(outputs) == word_i - 1:
            accuracy += 1
    return accuracy

def network_train(sizes, learning_rate):
    network = nw.Network(sizes)
    train_time = 0
    train_up = 1000
    error_result = 0
    print("train begin, size=%s, learning_rate=%f" % (str(sizes), learning_rate))
    while train_time < train_up:
        for word_index in range(1, 15):
            expect_output = np.zeros(14)
            expect_output[word_index - 1] = 1
            for img_index in range(240):
                file_name = './TRAIN/' + str(word_index) + '/' + str(img_index) + '.bmp'
                inputs = np.array(image.pic(file_name))
                network.train(inputs, expect_output, learning_rate)
        train_time = train_time + 1
        error_val = error_eva(network)
        print("training: %d, accuracy: %d%%" % (train_time, error_val))
        if train_up - train_time < 10:
            error_result += error_val
    error_result = error_result / 10
    return {'error': error_result, 'network': network}

if __name__ == "__main__":

    train_result = network_train([784, 10, 14], 0.05)

    while True:
        words = "苟利国家生死以岂因祸福避趋之"
        word_i = input("enter fold index (1-14): ")
        img_i = input("enter img index (0-255): ")
        file_name = './TRAIN/' + word_i + '/' + img_i + '.bmp'
        inputs = np.array(image.pic(file_name))
        outputs = train_result['network'].compute(inputs)
        print("recognition result: %s" % words[max_index(outputs)])
