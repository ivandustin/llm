from experiments.learned_pool_norm_test import *
from experiments.speed_benchmark import compiled_training
from experiments.readout_experiments import learn
import optax
for task in ['position','language']:
 for activation in ['sigmoid','gelu']:
  free=Model(task,0,'norm_'+activation);affine=Model(task,0,'affine_norm_'+activation)
  a=dict(nnx.to_flat_state(nnx.state(free,nnx.Param)));b=dict(nnx.to_flat_state(nnx.state(affine,nnx.Param)))
  for key,v in a.items():np.testing.assert_array_equal(v,b[key])
  x=jnp.array(np.random.default_rng(0).integers(0,31,(4,32 if task=='language' else 16)))
  np.testing.assert_array_equal(free(x),affine(x))
  assert count(affine)-count(free)==(64 if task=='language' else 510)
  print('shared initialization/logits match, parameter count verified',task,activation,flush=True)
for activation in ['sigmoid','gelu']:
 m=Model('language',0,'affine_norm_'+activation);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,65,(5,8,32)));y=jnp.array(rng.integers(0,65,(5,8)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step full-state parity',activation,flush=True)
