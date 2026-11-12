from pathlib import Path
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

PREPARATION_VERSION = 'class-coverage-v3'


def retain_sample(previous, chunk, spec):
    combined = pd.concat([previous, chunk], ignore_index=True)
    if combined.empty:
        return combined
    cap = spec.get('class_cap')
    if cap is None:
        return combined.nsmallest(spec['max_rows'], '_priority')
    if not isinstance(cap, int) or cap < 3:
        raise ValueError('class_cap must be an integer >=3')
    # Equal inclusion probability within each class, without synthetic records.
    return pd.concat([g.nsmallest(cap, '_priority') for _, g in combined.groupby(spec['label'], sort=True)], ignore_index=True)


def stratified_partitions(y, seed):
    """Approximate 70/10/20 allocation without copying or discarding records.

    Singleton classes stay in training. Two-record classes have one training
    and one test record, with no validation support. For n>=3, each partition
    receives at least one record. These constraints can alter exact split ratios.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == label))
        n = len(ids)
        if n == 1:
            train.extend(ids)
            continue
        if n == 2:
            train.append(ids[0]); test.append(ids[1])
            continue
        n_val = max(1, int(round(.1*n)))
        n_test = max(1, int(round(.2*n)))
        n_train = n-n_val-n_test
        train.extend(ids[:n_train])
        val.extend(ids[n_train:n_train+n_val])
        test.extend(ids[n_train+n_val:])
    return tuple(rng.permutation(np.asarray(part, dtype='int64')) for part in (train, val, test))


def prepare(spec, output, seed=42):
    """Sample real records, create disjoint partitions, and fit train-only transforms."""
    paths = sorted({p for pattern in spec['csv_globs'] for p in Path('/').glob(pattern.lstrip('/'))
                    if p.is_file() and not p.name.startswith('._')})
    if not paths:
        raise FileNotFoundError(f"No CSVs match {spec['csv_globs']}")
    rng = np.random.default_rng(seed)
    sample = None
    sources = []
    source_counts = {}
    for file_number, path in enumerate(paths, 1):
        print(f'File {file_number}/{len(paths)}: checksumming {path}', flush=True)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        sources.append({'path': str(path), 'sha256': digest.hexdigest()})
        offset = 0
        print(f'Sampling {path.name}', flush=True)
        for chunk in pd.read_csv(path, chunksize=50000):
            chunk.columns = chunk.columns.str.strip()
            chunk['_row_id'] = [f'{path}:{i}' for i in range(offset, offset + len(chunk))]
            offset += len(chunk)
            chunk['_priority'] = rng.random(len(chunk))
            chunk = chunk.dropna(subset=[spec['label']])
            chunk[spec['label']] = chunk[spec['label']].astype(str).str.strip()
            for label, count in chunk[spec['label']].value_counts().items():
                source_counts[label] = source_counts.get(label, 0) + int(count)
            sample = retain_sample(sample, chunk, spec)
            if offset % 1000000 == 0:
                print(f'{path.name}: {offset:,} rows scanned; {len(sample):,} retained', flush=True)
    if sample is None or sample.empty:
        raise ValueError('The configured CSV files contained no usable labeled rows.')
    sample = sample.dropna(subset=[spec['label']]).reset_index(drop=True)
    retained_counts_before_dedup = sample[spec['label']].value_counts().sort_index().to_dict()
    group_col, time_col = spec.get('group_column'), spec.get('time_column')
    excluded = set(spec['exclude_columns']) | {spec['label'], '_row_id', '_priority'}
    excluded.update(c for c in (group_col, time_col) if c)
    names = [c for c in sample.columns if c not in excluded]
    x = sample[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Keep identical feature vectors in one partition by removing duplicates before splitting.
    hashes = pd.util.hash_pandas_object(x, index=False)
    label_counts = pd.DataFrame({'h': hashes, 'y': sample[spec['label']]}).groupby('h').y.nunique()
    ambiguous_hashes = label_counts.index[label_counts.gt(1)]
    conflicting = int(len(ambiguous_hashes))
    conflict_policy = spec.get('conflicting_duplicate_policy', 'error')
    if conflict_policy not in {'error', 'drop_all'}:
        raise ValueError("conflicting_duplicate_policy must be 'error' or 'drop_all'")
    if conflicting and conflict_policy == 'error':
        raise ValueError(f'{conflicting} identical feature vectors have conflicting labels; audit before training.')

    ambiguous = hashes.isin(ambiguous_hashes) if conflicting else pd.Series(False, index=hashes.index)
    ambiguous_rows_removed = int(ambiguous.sum())
    # Ambiguous vectors cannot support a learnable mapping. Remove every row in
    # each conflicting group before splitting, then deduplicate the remainder.
    non_ambiguous_duplicate = hashes[~ambiguous].duplicated()
    keep = ~ambiguous
    keep.loc[~ambiguous] = ~non_ambiguous_duplicate
    duplicate_count = int(non_ambiguous_duplicate.sum())
    sample, x = sample.loc[keep].reset_index(drop=True), x.loc[keep].reset_index(drop=True)
    labels = sample[spec['label']].astype(str).str.strip()
    classes = sorted(labels.unique())
    y = pd.Categorical(labels, categories=classes).codes.astype('int64')
    ids = np.arange(len(x))
    protocol = spec['split']
    if protocol == 'group':
        if not group_col or sample[group_col].isna().any():
            raise ValueError('Group split needs a complete, meaningful group column.')
        groups = sample[group_col].astype(str).to_numpy()
        tv, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(ids, y, groups))
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.125, random_state=seed).split(tv, y[tv], groups[tv]))
        train, val = tv[a], tv[b]
    elif protocol == 'temporal':
        times = pd.to_datetime(sample[time_col], errors='raise', utc=True)
        boundaries = np.sort(times.unique())
        if len(boundaries) < 10:
            raise ValueError('Too few distinct timestamps for temporal evaluation.')
        first, second = boundaries[int(.7*len(boundaries))], boundaries[int(.8*len(boundaries))]
        train, val, test = ids[times < first], ids[(times >= first) & (times < second)], ids[times >= second]
    elif protocol == 'stratified':
        train, val, test = stratified_partitions(y, seed)
    else:
        raise ValueError(protocol)
    if any(len(z) == 0 for z in (train, val, test)) or set(y[train]) != set(range(len(classes))):
        raise ValueError('Empty partition or classes absent from training; inspect split.')
    names = [c for c in names if x.iloc[train][c].notna().any()]
    x = x[names]
    median = x.iloc[train].median()
    lower, upper = x.iloc[train].quantile(.001), x.iloc[train].quantile(.999)
    clean = x.fillna(median).clip(lower=lower, upper=upper, axis=1)
    scaler = StandardScaler().fit(clean.iloc[train])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, idx in [('train', train), ('val', val), ('test', test)]:
        arrays['x_'+name] = scaler.transform(clean.iloc[idx]).astype('float32')
        arrays['y_'+name] = y[idx]
        arrays['ids_'+name] = sample['_row_id'].iloc[idx].to_numpy(dtype=str)
    np.savez_compressed(output/'data.npz', **arrays)
    joblib.dump({'median': median, 'lower': lower, 'upper': upper, 'scaler': scaler, 'features': names}, output/'preprocessor.joblib')
    metadata = {'preparation_version': PREPARATION_VERSION, 'spec': spec, 'seed': seed, 'sources': sources, 'features': names, 'classes': classes,
                'duplicates_removed': duplicate_count, 'conflicting_duplicates': conflicting,
                'conflicting_duplicate_policy': conflict_policy,
                'ambiguous_rows_removed': ambiguous_rows_removed,
                'counts': {k: np.bincount(arrays['y_'+k], minlength=len(classes)).tolist() for k in ('train','val','test')}}
    metadata['missing_classes'] = {k: [classes[i] for i, count in enumerate(counts) if count == 0]
                                   for k, counts in metadata['counts'].items()}
    metadata['split_policy'] = ('Approximate 70/10/20; n=1 train only; n=2 train/test; n>=3 all partitions.'
                                if protocol == 'stratified' else protocol)
    metadata['source_class_counts'] = source_counts
    metadata['source_rows_scanned'] = int(sum(source_counts.values()))
    metadata['retained_class_counts_before_deduplication'] = retained_counts_before_dedup
    metadata['retained_class_counts_after_deduplication'] = labels.value_counts().sort_index().to_dict()
    metadata['partition_row_id_overlap'] = {
        'train_val': len(set(arrays['ids_train']) & set(arrays['ids_val'])),
        'train_test': len(set(arrays['ids_train']) & set(arrays['ids_test'])),
        'val_test': len(set(arrays['ids_val']) & set(arrays['ids_test'])),
    }
    metadata['sampling_protocol'] = ('class-capped; altered class priors; separate challenge evaluation'
                                     if spec.get('class_cap') is not None else 'uniform sample')
    metadata['minimum_heldout_support'] = min(min(metadata['counts'][k]) for k in ('val','test'))
    metadata['minimum_nonzero_heldout_support'] = min(
        count for key in ('val', 'test') for count in metadata['counts'][key] if count > 0
    )
    metadata['full_class_evaluation_supported'] = not any(metadata['missing_classes'].values())
    if not metadata['full_class_evaluation_supported']:
        print('WARNING: Some classes have no held-out support:', metadata['missing_classes'], flush=True)
        print('Pilot may proceed; full-class publication claims require more independent samples.', flush=True)
    # Publish completion metadata only after all artifacts have been written.
    temp = output/'metadata.tmp'
    temp.write_text(json.dumps(metadata, indent=2))
    os.replace(temp, output/'metadata.json')
    return metadata
