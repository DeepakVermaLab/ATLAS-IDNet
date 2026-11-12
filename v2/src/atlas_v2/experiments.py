import hashlib
import json
import os
import platform
import random
import socket
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .models import Detector, matched_width, parameter_count


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(y_true, proba, bins=15):
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if not count:
            continue
        bin_accuracy = float((predicted[mask] == y_true[mask]).mean())
        bin_confidence = float(confidence[mask].mean())
        weight = count / len(y_true)
        ece += weight * abs(bin_accuracy - bin_confidence)
        rows.append({
            'lower': float(lower),
            'upper': float(upper),
            'count': count,
            'accuracy': bin_accuracy,
            'confidence': bin_confidence,
            'absolute_gap': abs(bin_accuracy - bin_confidence),
        })
    return float(ece), rows


def score(y_true, predicted, classes, proba=None):
    class_count = len(classes)
    labels = list(range(class_count))
    metrics = {
        'accuracy': float(accuracy_score(y_true, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, predicted)),
        'precision_macro': float(precision_score(y_true, predicted, labels=labels, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, predicted, labels=labels, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, predicted, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, predicted, labels=labels, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, predicted)),
        'cohen_kappa': float(cohen_kappa_score(y_true, predicted, labels=labels)),
    }
    if proba is not None:
        one_hot = np.eye(class_count, dtype=np.float64)[y_true]
        metrics['log_loss'] = float(log_loss(y_true, proba, labels=labels))
        metrics['multiclass_brier'] = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
        metrics['ece_15'] = expected_calibration_error(y_true, proba, 15)[0]
        if set(labels) <= set(np.unique(y_true)):
            metrics['roc_auc_macro_ovr'] = float(
                roc_auc_score(y_true, proba, labels=labels, multi_class='ovr', average='macro')
            )
        else:
            metrics['roc_auc_macro_ovr'] = None

    benign = next((i for i, name in enumerate(classes) if str(name).lower() in {'benign', 'normal', 'normaltraffic'}), None)
    if benign is not None:
        true_attack = np.asarray(y_true) != benign
        predicted_attack = np.asarray(predicted) != benign
        tn = int((~true_attack & ~predicted_attack).sum())
        fp = int((~true_attack & predicted_attack).sum())
        fn = int((true_attack & ~predicted_attack).sum())
        tp = int((true_attack & predicted_attack).sum())
        metrics.update({
            'binary_intrusion_false_alarm_rate': float(fp / (fp + tn)) if fp + tn else None,
            'binary_intrusion_miss_rate': float(fn / (fn + tp)) if fn + tp else None,
        })
    return metrics


def resolve_device(requested='auto'):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but unavailable; using CPU.', flush=True)
        return torch.device('cpu')
    return torch.device(requested)


def probabilities(model, features, batch_size=4096, device=None):
    if not isinstance(model, torch.nn.Module):
        return model.predict_proba(features)
    device = device or next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(model(batch).softmax(1).cpu().numpy())
    return np.concatenate(outputs)


def linear_operation_estimate(model):
    macs = sum(module.in_features * module.out_features for module in model.modules() if isinstance(module, torch.nn.Linear))
    return {'linear_macs_per_flow': int(macs), 'linear_flops_per_flow': int(2 * macs)}


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _write_evaluation(out, split, y_true, proba, classes, ids):
    predicted = proba.argmax(1)
    metrics = score(y_true, predicted, classes, proba)
    atomic_json(out / f'{split}_metrics.json', metrics)
    np.savez_compressed(out / f'{split}_predictions.npz', y_true=y_true, proba=proba, ids=ids)
    report = classification_report(
        y_true,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out / f'{split}_per_class.csv')
    pd.DataFrame(
        confusion_matrix(y_true, predicted, labels=list(range(len(classes)))),
        index=classes,
        columns=classes,
    ).to_csv(out / f'{split}_confusion.csv')
    _, calibration = expected_calibration_error(y_true, proba, 15)
    pd.DataFrame(calibration).to_csv(out / f'{split}_calibration.csv', index=False)
    return metrics


def _baseline_factories(seed, threads):
    return {
        'random_forest': lambda: RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced_subsample',
            n_jobs=threads,
            random_state=seed,
        ),
        'extra_trees': lambda: ExtraTreesClassifier(
            n_estimators=300,
            class_weight='balanced',
            n_jobs=threads,
            random_state=seed,
        ),
        'hist_gradient': lambda: HistGradientBoostingClassifier(max_iter=250, random_state=seed),
        'logistic': lambda: LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            n_jobs=threads,
            random_state=seed,
        ),
    }


def run(
    data_dir,
    results_root,
    variant='atlas',
    seed=42,
    epochs=30,
    width=128,
    batch_size=512,
    patience=7,
    loss='balanced',
    threads=2,
    learning_rate=8e-4,
    weight_decay=2e-4,
    dropout=0.15,
    device='auto',
    capacity_target=None,
):
    """Train or resume one immutable experiment and evaluate validation/test sets."""
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / 'metadata.json').read_text())
    arrays_path = data_dir / 'data.npz'
    data_hash = sha256_file(arrays_path)
    source_hash = hashlib.sha256(
        b''.join(path.read_bytes() for path in sorted(Path(__file__).parent.glob('*.py')))
    ).hexdigest()
    classes = metadata['classes']
    if capacity_target is not None and variant not in _baseline_factories(seed, threads):
        width = matched_width(metadata['features'], len(classes), variant, capacity_target, seed, dropout)['width']

    config = {
        'variant': variant,
        'seed': seed,
        'epochs': epochs,
        'width': width,
        'batch_size': batch_size,
        'patience': patience,
        'loss': loss,
        'threads': threads,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'capacity_target': capacity_target,
        'selection_metric': 'validation_macro_f1',
        'data_sha256': data_hash,
        'source_sha256': source_hash,
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(results_root) / f'{variant}_{loss}_{seed}_{key}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('Reusing completed run:', out, flush=True)
        return out

    baseline_factories = _baseline_factories(seed, threads)
    actual_device = torch.device('cpu') if variant in baseline_factories else resolve_device(device)
    atomic_json(out / 'config.json', config)
    atomic_json(out / 'environment.json', {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'hostname': socket.gethostname(),
        'cpu_count': os.cpu_count(),
        'threads': threads,
        'requested_device': device,
        'actual_device': str(actual_device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)

    data = np.load(arrays_path)
    x_train, y_train = data['x_train'], data['y_train']
    started = time.perf_counter()
    if variant in baseline_factories:
        model = baseline_factories[variant]().fit(x_train, y_train)
        joblib.dump(model, out / 'model.joblib')
        atomic_json(out / 'architecture.json', {
            'family': type(model).__name__,
            'trainable_parameters': None,
            'capacity_matched': False,
        })
    else:
        model = Detector(metadata['features'], len(classes), variant, width, seed, dropout).to(actual_device)
        architecture = {
            'family': type(model).__name__,
            'variant': variant,
            'trainable_parameters': parameter_count(model),
            'groups': model.groups,
            'semantic_feature_map': model.feature_map,
            'capacity_target': capacity_target,
            **linear_operation_estimate(model),
        }
        atomic_json(out / 'architecture.json', architecture)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=learning_rate * 0.05,
        )
        history, best, stale, first = [], -1.0, 0, 0
        counts = np.bincount(y_train, minlength=len(classes)).astype(np.float64)
        weights = len(y_train) / (len(classes) * counts)
        weights = np.minimum(weights, np.percentile(weights, 95))
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=actual_device)
        checkpoint = out / 'last.pt'
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=actual_device, weights_only=True)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            _optimizer_to(optimizer, actual_device)
            scheduler.load_state_dict(state['scheduler'])
            history, best, stale, first = state['history'], state['best'], state['stale'], state['epoch'] + 1
            if 'torch_rng' in state:
                torch.set_rng_state(state['torch_rng'].cpu())
            print(f'Resuming {out.name} at epoch {first + 1} on {actual_device}', flush=True)

        for epoch in range(first, epochs):
            if stale >= patience:
                break
            epoch_started = time.perf_counter()
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(y_train))
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                if len(indices) == 1:
                    indices = np.concatenate([indices, order[:1]])
                optimizer.zero_grad(set_to_none=True)
                inputs = torch.from_numpy(x_train[indices]).to(actual_device)
                targets = torch.from_numpy(y_train[indices]).to(actual_device)
                logits = model(inputs)
                class_weights = weights if loss == 'balanced' else None
                per_row = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights, reduction='none')
                if loss == 'focal':
                    probability = logits.softmax(1).gather(1, targets[:, None]).squeeze(1)
                    per_row = per_row * (1.0 - probability).pow(2)
                objective = per_row.mean()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(objective.detach().cpu()) * len(indices)
            scheduler.step()

            validation_proba = probabilities(model, data['x_val'], batch_size, actual_device)
            validation_metrics = score(data['y_val'], validation_proba.argmax(1), classes, validation_proba)
            record = {
                'epoch': epoch + 1,
                'train_loss': total_loss / len(y_train),
                'val_macro_f1': validation_metrics['macro_f1'],
                'val_weighted_f1': validation_metrics['weighted_f1'],
                'val_mcc': validation_metrics['mcc'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_seconds': time.perf_counter() - epoch_started,
                'device': str(actual_device),
            }
            history.append(record)
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(out.name, record, flush=True)

            if record['val_macro_f1'] > best:
                best, stale = record['val_macro_f1'], 0
                torch.save(model.state_dict(), out / 'best.pt')
            else:
                stale += 1
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history,
                'best': best,
                'stale': stale,
                'epoch': epoch,
                'torch_rng': torch.get_rng_state(),
            }
            torch.save(state, out / 'last.tmp')
            os.replace(out / 'last.tmp', checkpoint)

        model.load_state_dict(torch.load(out / 'best.pt', map_location=actual_device, weights_only=True))

    validation_proba = probabilities(model, data['x_val'], batch_size, actual_device)
    test_proba = probabilities(model, data['x_test'], batch_size, actual_device)
    validation_metrics = _write_evaluation(
        out, 'validation', data['y_val'], validation_proba, classes, data['ids_val']
    )
    test_metrics = _write_evaluation(out, 'test', data['y_test'], test_proba, classes, data['ids_test'])
    elapsed = time.perf_counter() - started
    test_metrics['invocation_seconds'] = float(elapsed)
    test_metrics['model_file_bytes'] = int(sum(
        path.stat().st_size for path in out.glob('*') if path.suffix in {'.pt', '.joblib'}
    ))
    atomic_json(out / 'metrics.json', test_metrics)
    atomic_json(out / 'complete.json', {
        'config': config,
        'validation_metrics': validation_metrics,
        'metrics': test_metrics,
        'metadata_path': str(data_dir / 'metadata.json'),
    })
    return out


def load_model(data_dir, run_dir, device='cpu'):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    metadata = json.loads((Path(data_dir) / 'metadata.json').read_text())
    actual_device = resolve_device(device)
    torch.set_num_threads(config['threads'])
    if (run_dir / 'model.joblib').exists():
        return joblib.load(run_dir / 'model.joblib')
    model = Detector(
        metadata['features'],
        len(metadata['classes']),
        config['variant'],
        config['width'],
        config['seed'],
        config.get('dropout', 0.15),
    )
    model.load_state_dict(torch.load(run_dir / 'best.pt', weights_only=True, map_location=actual_device))
    return model.to(actual_device).eval()


def _process_memory_bytes():
    """Return Linux process RSS evidence without adding a runtime dependency."""
    status = Path('/proc/self/status')
    if not status.exists():
        return {}
    wanted = {'VmRSS': 'rss_bytes', 'VmHWM': 'peak_rss_bytes'}
    values = {}
    for line in status.read_text().splitlines():
        name, separator, raw = line.partition(':')
        if not separator or name not in wanted:
            continue
        fields = raw.split()
        if not fields:
            continue
        multiplier = 1024 if len(fields) == 1 or fields[1].lower() == 'kb' else 1
        values[wanted[name]] = int(fields[0]) * multiplier
    return values


def diagnostics(data_dir, run_dir, samples=256, repeats=100):
    """CPU timing, attention evidence and perturbation-based explanation checks."""
    run_dir = Path(run_dir)
    memory_before_load = _process_memory_bytes()
    model = load_model(data_dir, run_dir, 'cpu')
    memory_after_load = _process_memory_bytes()
    data = np.load(Path(data_dir) / 'data.npz')
    metadata = json.loads((Path(data_dir) / 'metadata.json').read_text())
    rng = np.random.default_rng(2026)
    selected = rng.choice(len(data['x_test']), min(samples, len(data['x_test'])), replace=False)
    x = data['x_test'][selected]

    rows = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for batch in (1, 16, 128, 512):
        features = data['x_test'][:min(batch, len(data['x_test']))]
        for _ in range(20):
            probabilities(model, features)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            probabilities(model, features)
            timings.append((time.perf_counter_ns() - start) / 1e6)
        rows.append({
            'batch': len(features),
            'p50_batch_ms': float(np.percentile(timings, 50)),
            'p95_batch_ms': float(np.percentile(timings, 95)),
            'p99_batch_ms': float(np.percentile(timings, 99)),
            'amortized_ms_per_flow': float(np.median(timings) / len(features)),
            'throughput_flows_s': float(len(features) * 1000 / np.median(timings)),
            'scope': 'prepared array to probabilities; raw parsing and reporting excluded',
            'device': 'cpu',
            'threads': 1,
        })
    torch.set_num_threads(previous_threads)
    pd.DataFrame(rows).to_csv(run_dir / 'latency.csv', index=False)

    started = time.perf_counter()
    original = probabilities(model, x)
    target = original.argmax(1)
    row_ids = np.arange(len(x))
    impact = np.empty_like(x)
    for feature in range(x.shape[1]):
        changed = x.copy()
        changed[:, feature] = 0
        impact[:, feature] = original[row_ids, target] - probabilities(model, changed)[row_ids, target]
    order = np.argsort(-impact, axis=1)
    curves = []
    for fraction in (0.1, 0.25, 0.5):
        count = max(1, int(x.shape[1] * fraction))
        top = order[:, :count]
        deleted = x.copy()
        deleted[row_ids[:, None], top] = 0
        retained = np.zeros_like(x)
        retained[row_ids[:, None], top] = x[row_ids[:, None], top]
        random_deleted = x.copy()
        for row in row_ids:
            random_deleted[row, rng.choice(x.shape[1], count, replace=False)] = 0
        curves.append({
            'fraction': fraction,
            'top_deletion_drop': float((original[row_ids, target] - probabilities(model, deleted)[row_ids, target]).mean()),
            'random_deletion_drop': float((original[row_ids, target] - probabilities(model, random_deleted)[row_ids, target]).mean()),
            'sufficiency_gap': float((original[row_ids, target] - probabilities(model, retained)[row_ids, target]).mean()),
        })
    pd.DataFrame(curves).to_csv(run_dir / 'faithfulness.csv', index=False)

    noisy = x + rng.normal(0, 0.01, x.shape).astype('float32')
    noisy_probability = probabilities(model, noisy)
    noisy_impact = np.empty_like(x)
    for feature in range(x.shape[1]):
        changed = noisy.copy()
        changed[:, feature] = 0
        noisy_impact[:, feature] = noisy_probability[row_ids, target] - probabilities(model, changed)[row_ids, target]
    count = max(1, int(0.25 * x.shape[1]))
    noisy_top = np.argsort(-noisy_impact, axis=1)[:, :count]
    overlap = [len(set(order[row, :count]) & set(noisy_top[row])) / count for row in row_ids]
    atomic_json(run_dir / 'stability.json', {
        'noise_std_standardized': 0.01,
        'top_quarter_jaccard_like_overlap': float(np.mean(overlap)),
        'prediction_agreement': float(np.mean(noisy_probability.argmax(1) == target)),
    })

    if isinstance(model, torch.nn.Module):
        with torch.inference_mode():
            _, attention = model(torch.from_numpy(x), return_attention=True)
        branch_names = ['traffic', 'protocol', 'temporal'] if attention.shape[1] == 3 else [f'branch_{i}' for i in range(attention.shape[1])]
        pd.DataFrame(attention.numpy(), columns=branch_names).to_csv(run_dir / 'attention_rows.csv', index=False)
        pd.DataFrame({
            'branch': branch_names,
            'mean': attention.mean(0).numpy(),
            'std': attention.std(0, unbiased=False).numpy(),
        }).to_csv(run_dir / 'attention_summary.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(5, 3))
    axis.plot([row['fraction'] for row in curves], [row['top_deletion_drop'] for row in curves], '-o', label='Ranked features')
    axis.plot([row['fraction'] for row in curves], [row['random_deletion_drop'] for row in curves], '-s', label='Random control')
    axis.set(xlabel='Fraction removed', ylabel='Predicted-class probability drop')
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'faithfulness.pdf')
    fig.savefig(run_dir / 'faithfulness.png', dpi=600)
    plt.close(fig)

    np.savez_compressed(
        run_dir / 'occlusion_evidence.npz',
        impact=impact,
        x=x,
        target=target,
        test_indices=selected,
        feature_names=np.asarray(metadata['features']),
    )
    architecture = json.loads((run_dir / 'architecture.json').read_text())
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    if isinstance(model, torch.nn.Module):
        parameter_bytes = int(sum(item.numel() * item.element_size() for item in model.parameters()))
        buffer_bytes = int(sum(item.numel() * item.element_size() for item in model.buffers()))
    else:
        parameter_bytes = None
        buffer_bytes = None
    memory_after_diagnostics = _process_memory_bytes()
    before_rss = memory_before_load.get('rss_bytes')
    after_rss = memory_after_load.get('rss_bytes')
    atomic_json(run_dir / 'resource_profile.json', {
        'scope': 'single CPU diagnostics process; prepared arrays to probabilities',
        'device': 'cpu',
        'threads': 1,
        'process_rss_before_model_load_bytes': before_rss,
        'process_rss_after_model_load_bytes': after_rss,
        'process_rss_delta_model_load_bytes': (
            max(0, after_rss - before_rss)
            if before_rss is not None and after_rss is not None
            else None
        ),
        'process_rss_after_diagnostics_bytes': memory_after_diagnostics.get('rss_bytes'),
        'process_peak_rss_after_diagnostics_bytes': memory_after_diagnostics.get('peak_rss_bytes'),
        'model_parameter_bytes': parameter_bytes,
        'model_buffer_bytes': buffer_bytes,
        'model_file_bytes': int(metrics['model_file_bytes']),
        'trainable_parameters': int(architecture['trainable_parameters']),
        'linear_macs_per_flow': int(architecture['linear_macs_per_flow']),
        'linear_flops_per_flow': int(architecture['linear_flops_per_flow']),
        'largest_profiled_input_batch_bytes': int(
            data['x_test'][:min(512, len(data['x_test']))].nbytes
        ),
        'limitations': [
            'Process RSS includes the Python and PyTorch runtime, loaded data, and prior allocations.',
            'VmHWM is a process high-water mark and is not attributable only to this model.',
            'Parameter and buffer byte counts are exact tensor storage, excluding allocator overhead.',
        ],
    })
    atomic_json(run_dir / 'explanation_scope.json', {
        'samples': len(x),
        'seconds': time.perf_counter() - started,
        'baseline': 'zero in training-standardized feature space',
        'method': 'single-feature occlusion with random-deletion control',
        'limitations': 'Perturbations can be off-distribution; no causal claim.',
    })


def summarize(root):
    root = Path(root)
    rows = []
    for path in root.rglob('complete.json'):
        payload = json.loads(path.read_text())
        rows.append({**payload['config'], **payload['metrics'], 'path': str(path.parent)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No completed experiments.')
    frame.to_csv(root / 'all_runs.csv', index=False)

    summaries = []
    group_columns = ['variant', 'loss', 'data_sha256', 'epochs', 'width']
    metric_columns = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'cohen_kappa', 'ece_15']
    for key, group in frame.groupby(group_columns, dropna=False):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            count = len(values)
            half_width = float(t.ppf(0.975, count - 1) * values.std(ddof=1) / np.sqrt(count)) if count > 1 else None
            summaries.append({
                **dict(zip(group_columns, key)),
                'metric': metric,
                'n': count,
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if count > 1 else None,
                'ci95_half_width': half_width,
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / 'seed_summary.csv', index=False)

    paired = []
    main = frame[frame.variant == 'atlas']
    for (loss_name, data_hash), reference in main.groupby(['loss', 'data_sha256']):
        for variant, comparison in frame[(frame.loss == loss_name) & (frame.data_sha256 == data_hash)].groupby('variant'):
            if variant == 'atlas':
                continue
            joined = reference.merge(comparison, on='seed', suffixes=('_atlas', '_comparison'))
            for metric in ('macro_f1', 'weighted_f1', 'mcc'):
                left = joined[f'{metric}_atlas'].to_numpy(dtype=float)
                right = joined[f'{metric}_comparison'].to_numpy(dtype=float)
                if len(left) < 2:
                    continue
                try:
                    wilcoxon_p = float(wilcoxon(left, right).pvalue) if np.any(left != right) else 1.0
                except ValueError:
                    wilcoxon_p = None
                paired.append({
                    'loss': loss_name,
                    'data_sha256': data_hash,
                    'comparison': variant,
                    'metric': metric,
                    'paired_seeds': len(left),
                    'mean_difference_atlas_minus_comparison': float(np.mean(left - right)),
                    'paired_t_pvalue': float(ttest_rel(left, right).pvalue),
                    'wilcoxon_pvalue': wilcoxon_p,
                })
    pd.DataFrame(paired).to_csv(root / 'paired_tests.csv', index=False)
    return summary
