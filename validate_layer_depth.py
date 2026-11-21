"""Evaluate the selected depth configuration on a fresh generated sample.

Architecture selection uses only the earlier 4,096-example evaluation sample.
This script reloads final weights and does not train or change hyperparameters.
"""
import json
from pathlib import Path
from benchmark import nnx,np,jnp,count
from budget_sweep import BatchedCandidate
from capabilities import ProbeTransformer,generate
from readout_experiments import metrics


def load_weights(model,path):
    flat=nnx.to_flat_state(nnx.state(model,nnx.Param))
    names=['/'.join(map(str,k)) for k,_ in flat]
    with np.load(path,allow_pickle=False) as arrays:
        assert set(arrays.files)==set(names),'Checkpoint structure differs from model'
        for (key,var),name in zip(flat,names):
            value=arrays[name]
            assert value.shape==var.get_value().shape
            var.set_value(jnp.array(value))
    nnx.update(model,nnx.from_flat_state(flat))


def main():
    summary=json.loads(Path('layer-depth-summary.json').read_text())
    best=max(summary['rows'],key=lambda r:r['accuracy_mean'])
    depth,width,hidden=best['hidden_layers'],best['width'],best['hidden']
    sample_seed=20261121
    n=16384
    x,y,_=generate('indexed_copy',16,n,np.random.default_rng(sample_seed))
    old_x,old_y,_=generate('indexed_copy',16,4096,np.random.default_rng(700016))
    original=json.loads(Path('layer-depth-results.json').read_text())
    results=[]
    for seed in range(3):
        model=BatchedCandidate(16,96,width,hidden,seed,depth)
        root=(f'budget-seed-{seed}' if depth==1 else f'depth-budget-seed-{seed}' if depth==2
              else f'stack-depth-{depth}-seed-{seed}')
        load_weights(model,Path(root)/f'candidate-width-{width}-seed-{seed}.npz')
        previous=next(r for r in original['runs'] if r['model']=='candidate' and r['hidden_layers']==depth and r['width']==width and r['seed']==seed)
        replicated=metrics(model,old_x,old_y)
        assert np.isclose(replicated['accuracy'],previous['final']['accuracy'],atol=1e-7)
        assert np.isclose(replicated['loss'],previous['final']['loss'],atol=1e-5)
        results.append(dict(model='selected_candidate',seed=seed,parameters=count(model),final=metrics(model,x,y)))
        ref=ProbeTransformer(96,16,16,nnx.Rngs(seed),depth=1,ff_hidden=162)
        load_weights(ref,Path(f'budget-seed-{seed}')/f'transformer-width-16-seed-{seed}.npz')
        results.append(dict(model='transformer',seed=seed,parameters=count(ref),final=metrics(ref,x,y)))
    aggregates=[]
    for name in ['selected_candidate','transformer']:
        vals=[100*r['final']['accuracy'] for r in results if r['model']==name]
        aggregates.append(dict(model=name,accuracy_mean=float(np.mean(vals)),accuracy_sd=float(np.std(vals,ddof=1)),seed_accuracies=vals))
    output=dict(selected=best,evaluation_seed=sample_seed,examples=n,
        selection_note='Best mean from prior grid; no fresh-sample tuning or further training.',runs=results,summary=aggregates)
    Path('layer-depth-fresh-validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
