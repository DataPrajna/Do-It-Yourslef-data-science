import h5py
import json
import numpy as np

class H5Writer:
    @staticmethod
    def write(to_filename, config, params_dict):
        with h5py.File(to_filename, 'w') as fid:
            meta_group = fid.create_group('meta')
            meta_group.attrs['config'] = json.dumps(config)
            data_group = fid.create_group('data')
            for key in params_dict:
                data_group.create_dataset(name=key, data=params_dict[key])


class H5Reader:
    @staticmethod
    def read(from_filename):
        with h5py.File(from_filename, 'r') as fid:
            config = json.loads(fid['meta'].attrs['config'])
            params_dict = dict()
            for key in  fid['data']:
                params_dict[key] = fid['data'][key].value
        return config, params_dict


