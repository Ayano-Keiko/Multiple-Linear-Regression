import numpy
import json
import os
import math
import matplotlib.pyplot as plt

class MLR:
    def __init__(self, logfile):
        self.slope = []
        self.intercept = 0
        self.saveDIR = f'{logfile}'
        self.iteration = 0
        self.learning_rate = 0
        self.RMSEs = []

    def fit(self, X, Y, iteration, learn_rate=0.01):
        '''
            Mutiply Linear Regression Implementation
            :param X: independence values, multi-dimension array
            :param Y: dependence value - 1-D array
            :param learn_rate: learning rate
            :return: (slopes, intercept)
            '''

        self.iteration = iteration
        self.learning_rate = learn_rate

        length = X.shape[0]  # length of independence values or dependence value
        dimension = X.shape[1]  # length of inputs columns
        # initialize Slope and Intercept
        self.slope = numpy.zeros(dimension)
        self.intercept = 0
        MSEs = []

        if not os.path.exists(self.saveDIR):
            os.mkdir(self.saveDIR)

        for iter in range(iteration):
            MSE = 0  # MSE in each iteration
            slopePartialDerivative = numpy.zeros(dimension)
            interceptPartialDerivative = 0

            for i in range(length):
                predict = self.intercept  # h(x)

                # calculate h(x)
                for j in range(dimension):
                    predict += self.slope[j] * X[i][j]

                # error
                error = Y[i] - predict
                MSE += error ** 2

                # calculate slope and intercept
                for j in range(dimension):
                    slopePartialDerivative[j] += error * X[i][j]
                interceptPartialDerivative += error

            MSE /= 2 * length
            MSEs.append(MSE)
            self.RMSEs.append(math.sqrt(MSE))

            for j in range(dimension):
                slopePartialDerivative[j] = slopePartialDerivative[j] / length * -1
            interceptPartialDerivative = interceptPartialDerivative / length * -1

            # update slope and intercept
            for j in range(dimension):
                self.slope[j] -= learn_rate * slopePartialDerivative[j]
            self.intercept -= learn_rate * interceptPartialDerivative

            # write log file of each iteration - DEBUG

            with open(f'{self.saveDIR}/MLRTraining[{iter + 1}]-[{learn_rate}]MSE', 'w', encoding='UTF-8') as fp:
                fp.write(str(MSE))

        # write log file of parameter
        with open(f'{self.saveDIR}/MLRModelParameters.json', 'w+', encoding='UTF-8') as jsn:
            parameter = {
                'learning rate': learn_rate,
                'iteration': iteration,
                'slope': self.slope.tolist(),
                'intercept': self.intercept,
                'MSE': MSEs[-1],
                'RMSE': self.RMSEs[-1]
            }
            json.dump(parameter, jsn)

        return self.slope, self.intercept

    def predict(self, values):
        predicts = []

        for i in range(values.shape[0]):
            p = self.intercept
            for j in range(values.shape[1]):
                p += self.slope[j] * values[i][j]

            predicts.append(float(p))

        return numpy.array(predicts)

    def drawMSEplot(self):
        mse = []
        index = [i + 1 for i in range(self.iteration)]
        for i in range(self.iteration):
            filename = f'{self.saveDIR}/MLRTraining[{i + 1}]-[{self.learning_rate}]MSE'
            mse.append(float(open(filename, 'r', encoding='UTF-8').read()))

        plt.figure()
        plt.title(f'MSR lr: {self.learning_rate} iteration: {self.iteration}')
        plt.plot(index, mse)
        plt.savefig(f'MSE lr{self.learning_rate}.png')

    def drawErrorBar(self, x, y):

        y_predict = []

        for i in range(x.shape[0]):
            p = self.intercept
            for j in range(x.shape[1]):
                p += x.iloc[i, j] * self.slope[j]
            y_predict.append(p)
        y_predict = numpy.array(y_predict)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f'Error Bar lr: {self.learning_rate}')
        ax.scatter(x.iloc[:, 0], x.iloc[:, 1], y_predict, color='#FF0000')
        ax.scatter(x.iloc[:, 0], x.iloc[:, 1], y, color='#0000FF')

        for i in range(x.shape[0]):
            ax.plot([x.iloc[i, 0], x.iloc[i, 0]], [x.iloc[i, 1], x.iloc[i, 1]], [y_predict[i], y[i]], color='#000000')
        plt.savefig(f'ErrorBar lr{self.learning_rate}.png')

    def drawRMSEPlot(self):
        index = [i + 1 for i in range(len(self.RMSEs))]

        plt.figure()
        plt.title(f'RMSE learn-rate{self.learning_rate}')
        plt.xlabel('iteration')
        plt.ylabel('RMSE')
        plt.plot(index, self.RMSEs)
        plt.savefig('RMSE.png')

    def r_square(self, y_actual, y_predict):
        mean = numpy.mean(y_actual)
        sumoferror = 0
        sumofmean = 0

        for index, actual_vaue in enumerate(y_actual):

            sumoferror += (actual_vaue - y_predict[index]) ** 2
            sumofmean += (actual_vaue - mean) ** 2

        return 1 - sumoferror / sumofmean

    def get_parameter(self):
        parameter = json.load(open(f'{self.saveDIR}/MLRModelParameters.json', 'r+', encoding='UTF-8'))

        return parameter
