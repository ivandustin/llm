"""Merge independently completed context-64 seed jobs with short-context runs."""
import json
from itertools import product
from pathlib import Path


def main():
    original=json.loads(Path('capabilities-before-parallel.json').read_text())
    runs=[r for r in original['runs'] if r['context']==16]
    for seed in range(3):
        part=json.loads(Path(f'capabilities-part-{seed}.json').read_text())
        assert part['config']['steps']==3000 and part['config']['width']==16
        assert all(r['context']==64 and r['seed']==seed for r in part['runs'])
        runs.extend(part['runs'])
    keys=[(r['task'],r['context'],r['seed'],r['model']) for r in runs]
    expected=set(product(original['config']['tasks'],[16,64],range(3),['cube','transformer']))
    assert len(keys)==len(set(keys))==60 and set(keys)==expected, 'Some runs are missing or duplicated'
    assert all(r['history'][-1]['step']==3000 for r in runs)
    assert all(0 <= r['accuracy'] <= 1 for r in runs)
    config=original['config']|dict(models=['cube','transformer'],seed_start=0,
        execution_note='Context 16 serial; context 64 resumed by seed with affinities 0-5, 6-11, 12-17. Controls ran concurrently. Do not compare synthetic timings.')
    runs.sort(key=lambda r:(r['context'],r['task'],r['seed'],r['model']))
    Path('capabilities-results.json').write_text(json.dumps(dict(config=config,runs=runs),indent=2)+'\n')
    print('Merged and validated all 60 main capability runs.')
    paths=[Path(f'capabilities-one-block-{seed}.json') for seed in range(3)]
    if all(p.exists() for p in paths):
        parts=[json.loads(p.read_text()) for p in paths]
        control=[r for part in parts for r in part['runs']]
        keys=[(r['context'],r['seed'],r['model']) for r in control]
        assert len(keys)==len(set(keys))==6
        assert set(keys)==set(product([16,64],range(3),['transformer']))
        assert all(r['history'][-1]['step']==3000 for r in control)
        config=parts[0]['config']|dict(seeds=3,seed_start=0,output='capabilities-one-block.json')
        Path('capabilities-one-block.json').write_text(json.dumps(dict(config=config,runs=control),indent=2)+'\n')
        print('Merged and validated all six one-block control runs.')


if __name__=='__main__': main()
