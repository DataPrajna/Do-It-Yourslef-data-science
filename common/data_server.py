from abc import ABCMeta, abstractmethod, abstractproperty
import csv
import numpy as np
import pandas as pd

'''
class BaseDataServer(metaclass=ABCMeta):
    def __init__(self, source_name):
        self.fid = None
        self.source_name = source_name


    @abstractmethod
    def get_body(self):
        pass

    @abstractmethod
    def get_header(self):
        pass

    @abstractmethod
    def post(self, to_source):
        pass

    @abstractmethod
    def close(self):
        pass




class CsvDataServer(BaseDataServer):
    def __init__(self, source_name):
        super().__init__(source_name)
        self._read()

    def _read(self):
        try:
            self.fid = open(self.source_name, 'r')
        except IOError as e:
            print('IOerror'.format(e))
        else:
            data_stream = csv.reader(self.fid)
            datas = []
            for row in data_stream:
                datas.append(row)
            self.np_data = np.asarray(datas)
            self.close()

    def post(self, to_source):
        return 'posting data to_source'

    def close(self):
        self.fid.close()

    def get_header(self):
        return self.np_data[0, :]

    def get_body(self):
        assert self.np_data.shape[0] > 1
        return self.np_data[1:, :]

class PandaDataServer(BaseDataServer):
    def __init__(self, source_name='https://vincentarelbundock.github.io/Rdatasets/csv/MASS/UScereal.csv'):
        super().__init__(source_name)
        self.pd_data = pd.read_csv(source_name)

    def post(self, to_source):
        return 'posting data to_source'

    def close(self):
        self.fid.close()

    def get_header(self):
        return self.pd_data.columns

    def get_body(self):
        return self.pd_data.values
'''

class BatchDataServer:
    def __init__(self, X, Y, batch_size = None):
        self.X = X
        self.Y = Y
        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        self.epoch = 0;

    def __next__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, item):
        pass
    def __iter__(self):
        pass

    def next(self):
        if self.end >= self.X.shape[0]:
            lx = self.X[self.start:self.end-1, :]
            ly = self.Y[self.start:self.end-1, :]
            self.start = 0
            self.end = self.batch_size
            self.epoch += 1
        else:
            lx = self.X[self.start:self.end, :]
            ly = self.Y[self.start:self.end, :]
            self.start = self.end
            self.end += self.batch_size
        return lx, ly


def test_batch_data_server():
    X = np.ones((5, 3))
    Y = np.ones((5, 1))
    X[0, 0] = 1
    X[1,0]  = 2
    X[2,0]  = 3
    X[3,0]  = 4
    X[4,0]  = 5
    batch_data = BatchDataServer(X,Y,batch_size =2)

    while batch_data.epoch < 10:
        x, y = batch_data.next()
        print('epoch= {}\n, x={}\n,  y={}'.format(batch_data.epoch, x, y))



if __name__=='__main__':
     #A = PandaDataServer('https://vincentarelbundock.github.io/Rdatasets/csv/MASS/UScereal.csv')
     #print(A.get_header())
     #print(A.get_body())
     test_batch_data_server()
