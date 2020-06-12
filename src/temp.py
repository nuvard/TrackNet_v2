import pandas as pd
import sys
sys.path.append('..')
from preprocessing.preprocessing import StandartScale, MinMaxScale, \
    ToCylindrical, Normalize, DropShort, DropSpinningTracks, DropFakes, ToCartesian, \
    Compose, ToBuckets, ConstraintsNormalize
data = pd.read_csv('../data/200.csv')
trnf = Compose([
    ToCylindrical(drop_old=False),
    StandartScale(columns=['r', 'phi', 'z'])
])

data = trnf(data)
data.to_csv('../data/data_radial_2.csv')