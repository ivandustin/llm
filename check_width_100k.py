from width_100k_test import *
from speed_benchmark import compiled_training
from readout_experiments import learn
import optax
for arch,(width,hidden) in SPECS.items():
 for task in ['position','language']:
  m=Model(task,0,arch);assert count(m)<=100000
  vocab=65 if task=='language' else 96;context=32 if task=='language' else 16
  expected=(context+vocab)*width+(width+vocab+1)*hidden+vocab+5*(hidden*hidden+hidden)
  assert count(m)==expected
  print(arch,task,'width',width,'hidden',hidden,'params',count(m),flush=True)
 m=Model('position',0,arch);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,4,16)));y=jnp.array(rng.integers(0,16,(5,4)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step complete-state parity',arch,flush=True)
