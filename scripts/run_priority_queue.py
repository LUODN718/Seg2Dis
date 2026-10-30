"""Run the prioritized USC12K main, head, RTA and FLR experiments in order."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--usc_root', required=True, type=Path)
    parser.add_argument('--backbone_checkpoint', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    plan = read(ROOT / 'configs/experiments.json')
    rows = {row['id']: row for row in plan['experiments']}
    queue = plan['priority_queue']
    if len(queue) != len(set(queue)) or set(queue) - rows.keys():
        raise ValueError('Priority queue contains duplicate or undefined experiments')
    status_path = args.output / 'queue_status.json'
    results = []
    for index, experiment_id in enumerate(queue, 1):
        row = rows[experiment_id]
        experiment_root = args.output / 'experiments' / experiment_id
        save(status_path, {'state': 'running', 'index': index, 'count': len(queue),
                           'experiment_id': experiment_id, 'completed': results})
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_experiment.py'),
                   '--config', str(ROOT / row['config']), '--output', str(experiment_root),
                   '--backbone_checkpoint', str(args.backbone_checkpoint),
                   '--usc_root', str(args.usc_root), '--device', args.device]
        with (args.output / f'{experiment_id}.stdout.log').open('a', encoding='utf-8') as stdout, \
                (args.output / f'{experiment_id}.stderr.log').open('a', encoding='utf-8') as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        if result.returncode:
            save(status_path, {'state': 'failed', 'index': index, 'count': len(queue),
                               'experiment_id': experiment_id, 'exit_code': result.returncode,
                               'completed': results})
            raise RuntimeError(f'{experiment_id} failed; inspect its logs')
        done = read(experiment_root / 'complete.json')
        if done.get('experiment_id') != experiment_id or done.get('architecture') != plan['architecture']:
            raise ValueError(f'{experiment_id} completed with mismatched identity')
        results.append({'experiment_id': experiment_id,
                        'checkpoint_sha256': done['checkpoint_sha256'],
                        'metrics': str(experiment_root / 'metrics.json')})
        save(status_path, {'state': 'running', 'index': index, 'count': len(queue),
                           'experiment_id': experiment_id, 'completed': results})

    summary = {'complete': True, 'architecture': plan['architecture'],
               'queue': queue, 'experiments': results}
    save(args.output / 'summary.json', summary)
    save(args.output / 'complete.json', summary)
    save(status_path, {'state': 'complete', 'completed': results, 'count': len(queue)})


if __name__ == '__main__':
    main()
