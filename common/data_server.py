from abc import ABCMeta, abstractmethod, abstractproperty
import csv
import numpy as np
import pandas as pd
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


if __name__=='__main__':
     A = PandaDataServer('https://vincentarelbundock.github.io/Rdatasets/csv/MASS/UScereal.csv')
     print(A.get_header())
     print(A.get_body())
