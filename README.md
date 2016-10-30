# Do It Yourslef (DIY), data science
This project explores key aspects  of data science by  practical examples. At the end we are expecting to produce a book, Do It Yourslef (DIY), data science


>>> train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           >>>...               2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

            now lets instantiate the class

           >>> lr = LinearRegressor(lr = 0.01, num_epocs = 100, print_frequency = 5)

            now lets train the linear regressor on the constructor training dataset train_x and train_y

           >>> lr.train(train_X = train_X, train_Y = train_Y)

