import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from atlas_v2.data import prepare, stratified_partitions


def test_rare_class_split():
    y = np.repeat(np.arange(7), [1, 2, 3, 4, 10, 50, 100])
    parts = stratified_partitions(y, 42)
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(y)))
    assert all(not set(a) & set(b) for a,b in [(parts[0],parts[1]),(parts[0],parts[2]),(parts[1],parts[2])])
    assert set(y[parts[0]]) == set(y)
    assert 0 not in y[parts[1]] and 0 not in y[parts[2]]
    assert 1 not in y[parts[1]] and 1 in y[parts[2]]
    for part in parts:
        assert set(range(2,7)) <= set(y[part])
    assert all(np.array_equal(a,b) for a,b in zip(parts,stratified_partitions(y,42)))
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        frame=pd.DataFrame({'bytes':np.arange(len(y)), 'duration':np.arange(len(y))*.3, 'protocol':np.arange(len(y))%3,'label':y})
        frame.to_csv(root/'input.csv',index=False)
        meta=prepare({'csv_globs':[str(root/'input.csv')], 'max_rows':len(y), 'label':'label',
                      'exclude_columns':[], 'split':'stratified'},root/'prepared')
        assert meta['missing_classes']['test']==['0']
        assert meta['missing_classes']['val']==['0','1']
        assert not meta['full_class_evaluation_supported']
        data=np.load(root/'prepared/data.npz')
        assert all(np.isfinite(data['x_'+k]).all() for k in ('train','val','test'))


def test_ambiguous_duplicates_are_dropped_before_split():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        frame = pd.DataFrame({
            'bytes': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'duration': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'label': ['attack', 'benign'] + ['attack'] * 5 + ['benign'] * 5,
        })
        frame.to_csv(root/'input.csv', index=False)
        spec = {
            'csv_globs': [str(root/'input.csv')], 'max_rows': len(frame),
            'label': 'label', 'exclude_columns': [], 'split': 'stratified',
            'conflicting_duplicate_policy': 'drop_all',
        }
        meta = prepare(spec, root/'prepared')
        assert meta['conflicting_duplicates'] == 1
        assert meta['ambiguous_rows_removed'] == 2
        assert meta['conflicting_duplicate_policy'] == 'drop_all'
        assert sum(sum(values) for values in meta['counts'].values()) == 10


if __name__=='__main__':
    test_rare_class_split()
    test_ambiguous_duplicates_are_dropped_before_split()
    print('PASS: rare splits and explicit removal of source-label ambiguity')
