# %%
csi_dir = '../../data/csi'
pic_dir = '../../data/pic'

import glob
datlist_rx0 = sorted(glob.glob(csi_dir + '/*subj[02356]*rx0*'))
datlist_rx1 = sorted(glob.glob(csi_dir + '/*subj[02356]*rx1*'))
datlist_rx2 = sorted(glob.glob(csi_dir + '/*subj[02356]*rx2*'))

piclist_cam0 = sorted(glob.glob(pic_dir + '/*subj[02356]*cam0*/*'))
piclist_cam1 = sorted(glob.glob(pic_dir + '/*subj[02356]*cam1*/*'))
piclist_cam2 = sorted(glob.glob(pic_dir + '/*subj[02356]*cam2*/*'))
piclist_cam3 = sorted(glob.glob(pic_dir + '/*subj[02356]*cam2*/*'))

# %%
import pandas as pd

df_ = pd.DataFrame.from_dict({
    'piclist_cam0': piclist_cam0, 
    'piclist_cam1': piclist_cam1, 
    'piclist_cam2': piclist_cam2, 
    'piclist_cam3': piclist_cam3, 
})

df_['env'] = df_['piclist_cam0'].apply(lambda x: x.split('/')[-2].split('_')[0][-1])
df_['subj'] = df_['piclist_cam0'].apply(lambda x: x.split('/')[-2].split('_')[1][-1])
df_['group'] = df_['piclist_cam0'].apply(lambda x: x.split('/')[-2].split('_')[2][-1])
df_['angle'] = df_['piclist_cam0'].apply(lambda x: x.split('/')[-2].split('_')[3][-1])
df_['t'] = df_['piclist_cam0'].apply(lambda x: x.split('/')[-2].split('_')[-1][-1])

# %%
df_.sort_values(['env', 'subj', 'group', 'angle', 't'], inplace=True)
# df_

# %%
import pandas as pd

df = pd.DataFrame.from_dict({
    'datlist_rx0': datlist_rx0, 
    'datlist_rx1': datlist_rx1, 
    'datlist_rx2': datlist_rx2
})

# import pandas as pd

# df = pd.DataFrame.from_dict({'datlist_rx0': datlist_rx0})
# df1 = pd.DataFrame.from_dict({'datlist_rx1': datlist_rx1})

# %%
df['env'] = df['datlist_rx0'].apply(lambda x: x.split('/')[-1].split('_')[0][-1])
df['subj'] = df['datlist_rx0'].apply(lambda x: x.split('/')[-1].split('_')[1][-1])
df['group'] = df['datlist_rx0'].apply(lambda x: x.split('/')[-1].split('_')[2][-1])
df['angle'] = df['datlist_rx0'].apply(lambda x: x.split('/')[-1].split('_')[3][-1])
df['rx'] = df['datlist_rx0'].apply(lambda x: x.split('/')[-1].split('_')[4][-1])
df['t'] = df['datlist_rx0'].apply(lambda x: x.split('/')[-1].split('_')[5][-1])

# %%
df.sort_values(['env', 'subj', 'group', 'angle', 't'], inplace=True)
df

# %%
randomer = 13928

df_.iloc[randomer][['subj', 'env', 'angle', 't']], df.iloc[randomer // 100][['subj', 'env', 'angle', 't']]

# %%
import csiread
import numpy as np
import tqdm

csilist_rx0 = list()
csilist_rx1 = list()
csilist_rx2 = list()

for i in tqdm.trange(len(df)):

    _csi0 = csiread.Intel(df.iloc[i]['datlist_rx0'], nrxnum=3, ntxnum=3, if_report=False)
    _csi0.read()
    csiarr0 = _csi0.get_scaled_csi()
    csilist_rx0.extend(np.array_split(csiarr0, 100))
    
    _csi1 = csiread.Intel(df.iloc[i]['datlist_rx1'], nrxnum=3, ntxnum=3, if_report=False)
    _csi1.read()
    csiarr1 = _csi1.get_scaled_csi()
    csilist_rx1.extend(np.array_split(csiarr1, 100))
    
    _csi2 = csiread.Intel(df.iloc[i]['datlist_rx2'], nrxnum=3, ntxnum=3, if_report=False)
    _csi2.read()
    csiarr2 = _csi2.get_scaled_csi()
    csilist_rx2.extend(np.array_split(csiarr2, 100))

# %%
import pickle as pk
import os

csislice_dstrootdir = '/home/lscsc/caizhijie/0420-wamera-benchmark/data/frames/'

for i in tqdm.trange(len(df_)):
    # try:
    pk.dump([
        pk.load(open(df_.iloc[0]['piclist_cam0'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
        pk.load(open(df_.iloc[0]['piclist_cam1'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
        pk.load(open(df_.iloc[0]['piclist_cam2'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
        pk.load(open(df_.iloc[0]['piclist_cam3'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
        csilist_rx0[i], 
        csilist_rx1[i], 
        csilist_rx2[i]
    ], open(csislice_dstrootdir + ('_'.join(df_.iloc[i]['piclist_cam0'].split('/')[-3:])[:-4] + '.pk'), 'wb'))
    # except FileNotFoundError:
    #     if not os.path.exists(csislice_dstrootdir + '/'.join(df_.iloc[i]['piclist_cam0'].split('/')[-3:-2])):
    #         os.mkdir(csislice_dstrootdir + '/'.join(df_.iloc[i]['piclist_cam0'].split('/')[-3:-2]))
    #     if not os.path.exists(csislice_dstrootdir + '/'.join(df_.iloc[i]['piclist_cam0'].split('/')[-3:-1])):
    #         os.mkdir(csislice_dstrootdir + '/'.join(df_.iloc[i]['piclist_cam0'].split('/')[-3:-1]))
    #     pk.dump([
    #         pk.load(open(df_.iloc[0]['piclist_cam0'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
    #         pk.load(open(df_.iloc[0]['piclist_cam1'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
    #         pk.load(open(df_.iloc[0]['piclist_cam2'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
    #         pk.load(open(df_.iloc[0]['piclist_cam3'].replace('data/', 'data/openpose/')[:-3] + 'pk', 'rb')), 
    #         csilist_rx0[i], 
    #         csilist_rx1[i], 
    #         csilist_rx2[i]
    #     ], open(csislice_dstrootdir + ('_'.join(df_.iloc[i]['piclist_cam0'].split('/')[-3:])[:-4] + '.pk'), 'wb'))

# %%
# df_

# %%
# import pickle as pk

# pk.dump(df_, open('framewise1.pk', 'wb'))

# %%


# %%



