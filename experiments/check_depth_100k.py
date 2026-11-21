from experiments.depth_100k_test import *
from experiments.speed_benchmark import compiled_training
from experiments.readout_experiments import learn
import optax
for arch in ['eight','twelve','transformer']:
 for task in ['position','language']:
  m=Model(task,0,arch);assert count(m)<=100000
  print(arch,task,count(m),flush=True)
 m=Model('position',0,arch);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,4,16)));y=jnp.array(rng.integers(0,16,(5,4)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step full-state parity',arch,flush=True)
