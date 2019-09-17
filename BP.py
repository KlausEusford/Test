import random
import math
import numpy as np
import matplotlib.pyplot as plt

#BP网络

class Bp_net():
    def __init__(self, layer_cons, learningrate, accuracy):
        self.result = []
        self.learningrate = learningrate
        self.accuracy = accuracy
        self.layer_const = layer_cons
        self.layer_number = len(layer_cons)
        self.output_number = layer_cons[self.layer_number - 1]
        self.weights_matrix = self.init_weights_matrix()
        self.biases_matrix = self.init_biases_matrix()

#偏置矩阵(更新偏置)
    def init_biases_matrix(self):
        biases_matrix = []
        for i in range(len(self.layer_const) - 1):
            biases = []
            for j in range(self.layer_const[i + 1]):
                biases.append(0 - random.random())
            biases_matrix.append(biases)
        return biases_matrix

#权重矩阵（更新权重）
    def init_weights_matrix(self):
        weights_matrix=[]
        for i in range(len(self.layer_const) - 1):
            weights_row=[]
            for j in range(self.layer_const[i + 1]):
                weights_col=[]
                for k in range(self.layer_const[i]):
                    weights_col.append(random.random())
                weights_row.append(weights_col)
            weights_matrix.append(weights_row)
        return weights_matrix

#前向传播
    def forward(self, input_data):
        self.result = []
        self.result.append(input_data)
        hidden = np.dot(self.weights_matrix[0],input_data)
        for i in range(len(hidden)):
            hidden[i][0]+=self.biases_matrix[0][i]
        hidden=sigmoid(hidden)
        self.result.append(hidden)

        for i in range(self.layer_number-2):
            hidden = np.dot(self.weights_matrix[i+1],hidden)
            for j in range(len(hidden)):
                hidden[j][0] += self.biases_matrix[i+1][j]
            if i != self.layer_number-3:
                hidden=sigmoid(hidden)
            self.result.append(hidden)

#逆向传播,调整网络
    def backward(self, output_data):
        residual_matrix=[]   #残差矩阵
        for i in range(self.layer_number-1-1,-1,-1):
            residual_matrix_element=[]
            if i == self.layer_number-2:
                for j in range(self.output_number):
                    result = [-(output_data[j][0] - self.result[i+1][j][0])]
                    residual_matrix_element.append(result)
                residual_matrix.append(residual_matrix_element)
            else:
                derivative_result = []
                for j in range(len(self.result[i+1])):
                    element = [self.result[i+1][j][0]*(1-self.result[i+1][j][0])]
                    derivative_result.append(element)
                residual_matrix_element = np.dot(np.transpose(self.weights_matrix[i+1]),
                                                    residual_matrix[self.layer_number-3-i])
                for j in range(len(derivative_result)):
                    residual_matrix_element[j][0] *= derivative_result[j][0]
                residual_matrix.append(residual_matrix_element)

        for i in range(self.layer_number-1-1,-1,-1):
            for j in range(len(self.weights_matrix[i])):
                for k in range(len(self.weights_matrix[i][j])):
                    self.weights_matrix[i][j][k] -= self.learningrate*residual_matrix[self.layer_number-2-i][j][0] * \
                                                    self.result[i][k][0]
        for i in range(self.layer_number-1-1,-1,-1):
            for j in range(len(self.biases_matrix[i])):
                self.biases_matrix[i][j] -= self.learningrate * residual_matrix[self.layer_number-2-i][j][0]

    def predict(self, input_data):
        net.forward(input_data)
        return net.result[net.layer_number-1][0]
#训练模型
    def train(self, input_data, output_data, iterations):
        for i in range(iterations):
            Error = 0 #初始error为0
            for j in range(len(input_data) // self.layer_const[0]):
                learning_data = []
                expect_data = []
                for k in range(self.layer_const[0]):
                    learning_data.append(input_data[j * self.layer_const[0] + k])
                for k in range(self.output_number):
                    expect_data.append(output_data[j * self.layer_const[0] + k])
                self.forward(learning_data)
                self.backward(expect_data)
#计算误差
                for k in range(self.output_number):
                    Error+=abs(expect_data[k][0]-self.result[len(self.layer_const)-1][k][0])
            if self.accuracy > Error / len(input_data):
                print(Error / len(input_data))
                break
            else:
                print( i," 更新误差为",Error / len(input_data))

#sigmoid激活函数（隐含层）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


net = Bp_net([1, 10, 1], 0.1, 0.01)
#定义pai

input_x = []
input_y = []
#iterations 设置为1000,1500,2000,2500,5000
for i in range(2000):
    temp_x = [random.uniform(-np.pi, np.pi)] #随机生成一个介于pi和-pi的数
    input_x.append(temp_x)
    temp_y = [math.sin(temp_x[0])]
    input_y.append(temp_y)
net.train(input_x, input_y, 100)

predict = []
for j in range(len(input_x)):
    predict.append(net.predict([input_x[j]]))

plt.grid(True)   #显示网格
plt.scatter(input_x, input_y)    #拟合曲线
plt.scatter(input_x, predict)   #预期曲线
plt.show()
