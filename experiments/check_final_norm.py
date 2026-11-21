from experiments.final_norm_test import *
from experiments.speed_benchmark import compiled_training
from experiments.readout_experiments import learn
import optax
for task in ['position','language']:
 base=BatchedCandidate(32 if task=='language' else 16,65 if task=='language' else 96,64,12,0,6)
 plain=Model(task,0,'plain');final=Model(task,0,'final')
 old=dict(nnx.to_flat_state(nnx.state(base,nnx.Param)));new=dict(nnx.to_flat_state(nnx.state(final.base,nnx.Param)))
 for key,a in old.items():np.testing.assert_array_equal(a,new[key])
 x=jnp.array(np.random.default_rng(0).integers(0,65,(4,32 if task=='language' else 16)))
 np.testing.assert_array_equal(base(x),plain(x))
 assert count(final)==count(plain)+24
 print('identical shared initialization, baseline logits, +24 params:',task,flush=True)
fixed=Model('position',0,'fixed')
assert count(fixed)==9976
for variant in ['plain','final','fixed']:
 m=Model('position',0,variant);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,8,16)));y=jnp.array(rng.integers(0,16,(5,8)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step full-state parity:',variant,flush=True)
