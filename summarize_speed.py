"""Validate and format the isolated warmed-up speed comparison."""
import json
from pathlib import Path
import numpy as np


def main():
    r=json.loads(Path('speed-results.json').read_text())
    assert r['config']['trials']==7
    assert r['config']['parameters']==dict(candidate=9976,transformer=9970)
    labels=dict(inference_batch_1='Inference, batch 1',inference_batch_64='Inference, batch 64',
                python_training_batch_64='Python training, batch 64',compiled_training_batch_64='Compiled training, batch 64')
    lines=['Six-hidden-layer model versus one-block transformer; isolated CPU benchmark.',
           r['config']['cpu'],
           'Context 16; FP32; warmed up; seven alternating-order trials; median time.',
           'Both models have all 22 logical CPUs available. Generation, transfers and compilation excluded.',
           '', 'Measurement                     Yours ms    Transformer ms   Speedup']
    for mode,label in labels.items():
        vals=r['results'][mode]
        for v in vals.values():
            assert len(v['seconds_per_step_trials'])==7
            assert all(np.isfinite(x) and x>0 for x in v['seconds_per_step_trials'])
        a=vals['candidate']['milliseconds_per_step_median']
        b=vals['transformer']['milliseconds_per_step_median']
        assert np.isclose(b/a,r['candidate_speedup'][mode])
        lines.append(f'{label:32s}{a:8.3f}       {b:8.3f}       {b/a:5.2f}x')
    lines.extend(['', 'Inference synchronizes each request and includes JIT dispatch overhead.',
        'Python training synchronizes complete model and optimizer state at trial end.',
        'Compiled training uses lax.scan, removing Python dispatch between optimizer updates.',
        'Its five-step losses and all model/Adam-state leaves match ordinary training.',
        'Compilation/first-call timings and all individual trials are in speed-results.json.',
        'This is throughput/latency, not time to target accuracy. Transformer learned in fewer updates.',
        'Accuracy reference: selected model 98.15%, transformer 100% on the fresh evaluation sample.'])
    text='\n'.join(lines)+'\n'
    Path('speed-summary.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
