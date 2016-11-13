#test cases for linear regressor
from predictive_modelling_part_0.linear_regressor import  LinearRegressor
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt



def test_first_order_polynomial():
    lr = LinearRegressor(lr = 0.01,num_epocs = 1000,print_frequency = 100, num_features = 2)
    train_x = numpy.linspace(-1,1,100,dtype = numpy.float32)
    print(train_x)
    train_x = train_x.reshape(-1,1)
    train_y = 5 + 2*train_x
    #plt.plot(train_x, train_y, 'r.')
    lr.set_parameters()
    W = lr.train(train_X = train_x, train_Y = train_y)
    #plt.show()
    lr.set_parameters(W=W)
    y_hat =  lr.predict(X = train_x)
    y_hat = numpy.asarray(y_hat, dtype =numpy.float32, order = None)
    y_hat = y_hat.reshape(-1,1)
    plt.plot(train_x, y_hat, 'r.')
    plt.hold(True)
    plt.plot(train_x, train_y)
    plt.show()


def test_randomdata_first_order_polynomial():
    lr = LinearRegressor(lr = 0.01,num_epocs = 1000,print_frequency = 100, num_features = 2)
    num_points = 1000
    vectors_set = []
    for i in range(num_points):
        x1 = numpy.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + numpy.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])

    train_x = [v[0] for v in vectors_set]
    train_y = [v[1] for v in vectors_set]

    train_x = numpy.asarray(train_x)
    train_x = train_x.reshape(-1,1)
    train_y = numpy.asarray(train_y)
    train_y = train_y.reshape(-1,1)

    #train_x = numpy.linspace(-1,1,100,dtype = numpy.float32)
    # print(train_x)
    #train_x = train_x.reshape(-1,1)
    #train_y = 5 + 2*train_x
    #plt.plot(train_x, train_y, 'r.')
    lr.set_parameters()
    W = lr.train(train_X = train_x, train_Y = train_y)
    #plt.show()
    lr.set_parameters(W=W)
    y_hat =  lr.predict(X = train_x)
    y_hat = numpy.asarray(y_hat, dtype =numpy.float32, order = None)
    y_hat = y_hat.reshape(-1,1)
    plt.plot(train_x, train_y, 'go', label = 'Original Data')
    plt.legend()
    plt.hold(True)
    plt.plot(train_x, y_hat, 'b-')

    plt.show()




def test_multi_order_polynomial():
    lr = LinearRegressor(lr = 0.01, num_epocs = 1000, print_frequency = 100, num_features = 5)
    train_x = numpy.linspace(-1, 1, 1000000, dtype = numpy.float32)
    train_x = train_x.reshape(-1,1)
    train_y = 5  + 2*train_x + 3*train_x**2
    lr.set_parameters()
    W = lr.train(train_X = train_x, train_Y = train_y)
    W = lr.set_parameters(W = W)
    y_hat = lr.predict(X = train_x)
    y_hat = numpy.asarray(y_hat, dtype = numpy.float32, order = None)
    y_hat = y_hat.reshape(-1,1)
    plt.plot(train_x, y_hat, 'r.')
    plt.hold(True)
    plt.plot(train_x, train_y)
    plt.show()




def test_sinusoidal_regression():
    lr = LinearRegressor(lr = 0.001, num_epocs = 5000, print_frequency = 100, num_features = 5)
    train_x = numpy.linspace(-3,3, 10000, dtype = numpy.float32)
    train_x = train_x.reshape(-1,1)
    train_y = numpy.sin(train_x)
    lr.set_parameters()
    W = lr.train(train_X = train_x, train_Y = train_y)
    W = lr.set_parameters(W=W)
    y_hat = lr.predict(X = train_x)
    y_hat = numpy.asarray(y_hat, dtype = numpy.float32, order = None)
    y_hat = y_hat.reshape(-1,1)
    plt.plot(train_x, y_hat, 'r.')
    plt.hold(True)
    plt.plot(train_x, train_y)
    plt.show()





if __name__ == "__main__":
    #test_sinusoidal_regression()
    #test_multi_order_polynomial()
    #test_first_order_polynomial()
    test_randomdata_first_order_polynomial()
