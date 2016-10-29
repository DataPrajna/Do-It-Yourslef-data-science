from abc import ABCMeta, abstractmethod, abstractproperty
import csv


class BaseDataServer(metaclass=ABCMeta):
    def __init__(self):
        self.data_stream = None

    @abstractmethod
    def get(self, from_source):
        pass

    @abstractmethod
    def post(self, to_source):
        pass

    @abstractmethod
    def close(self):
        pass


class IoStringDataServer(BaseDataServer):
    def get(self, from_source):
        return 'reading from {}'.format(from_source)

    def post(self, to_source):
        return 'posting data to {}'.format(to_source)

    def close(self):
        pass


class CsvDataServer(BaseDataServer):
    def get(self, csv_file_name):
        try:
            self.fid = open(csv_file_name, 'r')
        except IOError as e:
            print('IOerror'.format(e))

        print('reading from {}'.format(csv_file_name))
        self.data_stream = csv.reader(self.fid)
        datas = []
        for row in self.data_stream:
            datas.append(row)

        return datas



    def post(self, to_source):
        return 'posting data to_source'

    def close(self):
        self.fid.close()



if __name__=='__main__':
     A = CsvDataServer()

     print(A.get('../data/SalesJan2009-1.csv'))