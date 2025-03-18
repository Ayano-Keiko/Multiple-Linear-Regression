import argparse
import os
import time
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from Linear_Regression import MLR
from Prepossessing.prepossess import DataPreparing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Multiple Linear Regression',
        description='self-made module to implementation Multiple Linear Regression algorithm',
    )
    parser.add_argument('-i', '--iteration', help='the number of iteration',
                        required=True, type=int)
    parser.add_argument('--learn_rate', help='learning rate',
                        required=True, type=float)

    args = parser.parse_args()

    log_dir = "logs-organic pollutants"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    iteration = args.iteration
    learning_rate = args.learn_rate

    df = pandas.read_excel('./data/myDataMLR.xlsx')

    numFeatures = df.shape[1]

    input_features = df.iloc[:, :numFeatures -1].values
    target_feature = df.iloc[:, numFeatures - 1].values

    df_scaling = pandas.DataFrame()

    for j in range(numFeatures - 1):
        df_scaling[j] = DataPreparing.z_score(df.iloc[:, j].values)

    # df_scaling.to_excel('./normalization.xlsx')
    x_train, x_test, y_train, y_test = train_test_split(df_scaling.values, target_feature, random_state=66)

    lr = MLR.MLR(log_dir)
    start = time.time()
    slope, intercept = lr.fit(x_train, y_train, iteration, learning_rate)
    end = time.time()

    # print parameter
    print(lr.get_parameter())

    # draw each MSE
    # lr.drawMSEplot()
    # lr.drawRMSEPlot()
    # draw error bar
    # lr.drawErrorBar(df_scaling, target_feature)

    y_predict = lr.predict(x_test)
    print("-"* 30)
    print(f"R square: {lr.r_square(y_test, y_predict)}")
    
    # predict custom values
    print("-" * 30)
    x_arr = []
    for j in range(numFeatures - 1):
        x_arr.append(float(input(f'please enter {df.columns[j]}: ')))

    x_predict = numpy.array([
        x_arr
    ])
    for j in range(numFeatures - 1):
        x_predict[:, j] = DataPreparing.z_score_predict(x_predict[:, j], input_features[:, j])

    y_user_predict = lr.predict(x_predict)
    print(f'the predict of input {x_arr} is: {y_user_predict[0]: .2f}')


    # only 2D -> 2 target
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_title(f'MLR 3D plot lr: {learning_rate} iteration: {iteration}')
    # ax.scatter(df_scaling[:, 0], df_scaling[:, 1], target_feature)
    # ax.plot_trisurf(df_scaling[:, 0], df_scaling[:, 1], y_predict, color='#FF0000')
    # ax.set_xlabel(df.columns[0])
    # ax.set_ylabel(df.columns[1])
    # ax.set_zlabel(df.columns[2])
    # fig.savefig('MLR.png')
    # plt.close(fig)

    # Program time
    print("-" * 30)
    print(f"Program Running Time: {end - start} s")
