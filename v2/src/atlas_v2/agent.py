import hashlib
import json
import time
from pathlib import Path

import pandas as pd


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class EvidenceAgent:
    """Tool-using evidence agent with deterministic claim verification.

    The agent plans which artifact tools are required, gathers their outputs,
    generates bounded findings and rejects every claim lacking a source artifact.
    It is deliberately language-model independent so its numeric fidelity can be
    evaluated without an external provider or unverifiable generation step.
    """

    TOOL_REQUIREMENTS = {
        'performance': ('metrics.json', 'test_per_class.csv', 'config.json'),
        'latency': ('latency.csv',),
        'explanation': ('faithfulness.csv', 'stability.json', 'attention_summary.csv'),
    }

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.trace = []

    def plan(self, question='complete'):  # question is persisted for protocol evidence.
        requested = ['performance', 'latency', 'explanation'] if question == 'complete' else [question]
        unknown = [name for name in requested if name not in self.TOOL_REQUIREMENTS]
        if unknown:
            raise ValueError(f'Unsupported evidence request: {unknown}')
        self.trace.append({'step': 'plan', 'question': question, 'tools': requested})
        return requested

    def _read_json(self, name):
        path = self.run_dir / name
        payload = json.loads(path.read_text())
        self.trace.append({'step': 'tool', 'tool': 'read_json', 'artifact': name, 'sha256': _sha256(path)})
        return payload, path

    def _read_csv(self, name):
        path = self.run_dir / name
        payload = pd.read_csv(path)
        self.trace.append({'step': 'tool', 'tool': 'read_csv', 'artifact': name, 'sha256': _sha256(path)})
        return payload, path

    def gather(self, tasks):
        missing = []
        for task in tasks:
            missing.extend(name for name in self.TOOL_REQUIREMENTS[task] if not (self.run_dir / name).exists())
        if missing:
            raise FileNotFoundError(f'Missing required evidence artifacts: {sorted(set(missing))}')

        evidence = {}
        if 'performance' in tasks:
            metrics, metrics_path = self._read_json('metrics.json')
            per_class, per_class_path = self._read_csv('test_per_class.csv')
            config, config_path = self._read_json('config.json')
            label_column = per_class.columns[0]
            class_rows = per_class[~per_class[label_column].astype(str).isin(['accuracy', 'macro avg', 'weighted avg'])]
            evidence['performance'] = {
                'metrics': metrics,
                'worst_recall': float(class_rows['recall'].min()) if 'recall' in class_rows else None,
                'sources': [str(metrics_path), str(per_class_path), str(config_path)],
                'variant': config['variant'],
                'seed': config['seed'],
            }
        if 'latency' in tasks:
            latency, latency_path = self._read_csv('latency.csv')
            single = latency.loc[latency.batch == 1].iloc[0]
            evidence['latency'] = {
                'p50_single_flow_ms': float(single.p50_batch_ms),
                'p95_single_flow_ms': float(single.p95_batch_ms),
                'maximum_measured_throughput_flows_s': float(latency.throughput_flows_s.max()),
                'scope': str(single.scope),
                'sources': [str(latency_path)],
            }
        if 'explanation' in tasks:
            faithfulness, faithfulness_path = self._read_csv('faithfulness.csv')
            stability, stability_path = self._read_json('stability.json')
            attention, attention_path = self._read_csv('attention_summary.csv')
            top_branch = attention.sort_values('mean', ascending=False).iloc[0]
            evidence['explanation'] = {
                'top_deletion_drop_25': float(faithfulness.loc[faithfulness.fraction == 0.25, 'top_deletion_drop'].iloc[0]),
                'random_deletion_drop_25': float(faithfulness.loc[faithfulness.fraction == 0.25, 'random_deletion_drop'].iloc[0]),
                'prediction_agreement_under_noise': float(stability['prediction_agreement']),
                'dominant_attention_branch': str(top_branch.branch),
                'dominant_attention_mean': float(top_branch['mean']),
                'sources': [str(faithfulness_path), str(stability_path), str(attention_path)],
            }
        return evidence

    def reason(self, evidence):
        findings = []
        if 'performance' in evidence:
            item = evidence['performance']
            metrics = item['metrics']
            findings.append({
                'claim_type': 'performance',
                'statement': (
                    f"The evaluated {item['variant']} run achieved macro-F1 {metrics['macro_f1']:.6f}, "
                    f"weighted-F1 {metrics['weighted_f1']:.6f}, and MCC {metrics['mcc']:.6f}."
                ),
                'values': {key: metrics[key] for key in ('macro_f1', 'weighted_f1', 'mcc')},
                'sources': item['sources'],
            })
        if 'latency' in evidence:
            item = evidence['latency']
            findings.append({
                'claim_type': 'latency',
                'statement': (
                    f"Single-flow CPU inference had p50 {item['p50_single_flow_ms']:.6f} ms and "
                    f"p95 {item['p95_single_flow_ms']:.6f} ms under the recorded timing scope."
                ),
                'values': {key: item[key] for key in ('p50_single_flow_ms', 'p95_single_flow_ms')},
                'sources': item['sources'],
            })
        if 'explanation' in evidence:
            item = evidence['explanation']
            findings.append({
                'claim_type': 'explanation',
                'statement': (
                    f"At 25% deletion, ranked-feature removal changed predicted-class probability by "
                    f"{item['top_deletion_drop_25']:.6f}, versus {item['random_deletion_drop_25']:.6f} "
                    f"for the random control; the dominant attention branch was "
                    f"{item['dominant_attention_branch']} ({item['dominant_attention_mean']:.6f})."
                ),
                'values': {key: item[key] for key in (
                    'top_deletion_drop_25',
                    'random_deletion_drop_25',
                    'dominant_attention_mean',
                )},
                'sources': item['sources'],
            })
        self.trace.append({'step': 'reason', 'finding_count': len(findings)})
        return findings

    def verify(self, findings):
        rejected = []
        for index, finding in enumerate(findings):
            missing = [path for path in finding.get('sources', []) if not Path(path).exists()]
            if not finding.get('sources') or missing:
                rejected.append({'finding': index, 'missing_sources': missing or ['no source supplied']})
        verdict = {'accepted': len(findings) - len(rejected), 'rejected': rejected, 'unsupported_claim_rate': len(rejected) / max(1, len(findings))}
        self.trace.append({'step': 'verify', **verdict})
        return verdict

    def run(self, output_dir=None, question='complete'):
        started = time.perf_counter()
        tasks = self.plan(question)
        evidence = self.gather(tasks)
        findings = self.reason(evidence)
        verification = self.verify(findings)
        report = {
            'agent': 'artifact-grounded finite-state evidence agent',
            'question': question,
            'findings': findings,
            'verification': verification,
            'tool_trace': self.trace,
            'latency_ms': (time.perf_counter() - started) * 1000,
            'limitations': [
                'The agent summarizes measured artifacts and does not establish causality.',
                'It does not replace domain-expert incident assessment.',
                'No language model was used; no LLM capability is claimed.',
            ],
        }
        output = Path(output_dir or self.run_dir)
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / 'agent_report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        os_replace = output / 'agent_report.json'
        temporary.replace(os_replace)
        return report


def evaluate_agent(run_dirs, output_dir):
    rows = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in map(Path, run_dirs):
        started = time.perf_counter()
        try:
            report = EvidenceAgent(run_dir).run(run_dir / 'agent')
            rows.append({
                'run_dir': str(run_dir),
                'complete': True,
                'finding_count': len(report['findings']),
                'unsupported_claim_rate': report['verification']['unsupported_claim_rate'],
                'agent_latency_ms': report['latency_ms'],
                'evaluation_seconds': time.perf_counter() - started,
            })
        except (FileNotFoundError, ValueError, KeyError, IndexError) as error:
            rows.append({
                'run_dir': str(run_dir),
                'complete': False,
                'error': str(error),
                'evaluation_seconds': time.perf_counter() - started,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'agent_evaluation.csv', index=False)
    return frame
