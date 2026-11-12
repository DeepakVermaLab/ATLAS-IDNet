import numpy as np
import pandas as pd
from atlas_v2.data import retain_sample

rng = np.random.default_rng(42)
frame = pd.DataFrame({'label': ['common']*1000+['rare']*37, 'row': range(1037), '_priority':rng.random(1037)})
spec = {'label':'label','class_cap':100,'max_rows':100}
sample = None
for idx in np.array_split(np.arange(len(frame)), 7):
    sample = retain_sample(sample, frame.iloc[idx], spec)
expected = retain_sample(None,frame,spec)
assert set(sample.row)==set(expected.row)
assert sample.row.is_unique
assert sample.label.value_counts().to_dict()=={'common':100,'rare':37}
print('PASS: class reservoir independent of chunking, rare rows retained, no duplication')
