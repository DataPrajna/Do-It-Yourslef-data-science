

import pandas as pd
import numpy as np

f = pd.DataFrame()
#LCC
names = ['VeNICE', 'CNNIQ', 'CORNIA', 'BRISQUE', 'SSIM', 'PSNR']
data = [[0.963, 0.975, 0.983, 0.96, 0.94, 0.964],
        [0.953, 0.981, 0.984, 0.953, 0.933, 0.953],
        [0.951, 0.965, 0.987, 0.968, 0.917, 0.935 ],
        [0.922,0.973,0.985,0.951,0.903,0.942],
        [0.921,0.955,0.982,0.893,0.939,0.906],
        [0.873,0.876,0.926,0.779,0.870,0.856]]


data = np.transpose(data)

data = np.asarray(data)

col = ['JP2K', 'JPEG', 'WN', 'GBLUR', 'FF', 'ALL']
df = pd.DataFrame(data,index=col,columns=names)
df.plot(kind='bar', fontsize=30, rot=0)
import matplotlib.pylab  as plt
plt.ylabel('LCC', fontsize=30)
plt.xlabel('Noise type ', fontsize=30)
plt.savefig('LCC.png')

data = [[0.956, 0.967, 0.982, 0.954, 0.94, 0.96],
        [0.952,0.977,0.978,0.962,0.908,0.956],
        [0.943,0.955,0.976,0.969,0.906,0.942],
        [0.914,0.965,0.979,0.951,0.877,0.940],
        [0.939,0.946,0.964,0.907,0.941,0.913],
        [0.870,0.885,0.942,0.763,0.874,0.866]]

data = np.transpose(data)

data = np.asarray(data)

col = ['JP2K', 'JPEG', 'WN', 'GBLUR', 'FF', 'ALL']
df = pd.DataFrame(data,index=col,columns=names)
df.plot(kind='bar', fontsize=30, rot=0)
import matplotlib.pylab  as plt
plt.ylabel('SROCC', fontsize=30)
plt.xlabel('Noise type ', fontsize=30)
plt.savefig('SROCC.png')

col = ['SSIM', 'VIF', 'LBIQ', 'BRISQUE', 'DLIQ', 'VeNICE']


data = [[0.815, 0.76, 0.74, 0.62, 0.84, 0.87]]

#data = np.transpose(data)

data = np.asarray(data)
df = pd.DataFrame(data,index=['Overall Average DMOS'],columns=col)
df.plot(kind='bar', fontsize=30, rot=0)
import matplotlib.pylab  as plt
plt.ylabel('SROCC', fontsize=30)
plt.savefig('SORCC.png')