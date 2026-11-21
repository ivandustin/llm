from experiments.sinusoidal_test import *
from experiments.benchmark import np
from experiments.readout_experiments import learn
from experiments.speed_benchmark import compiled_training
import optax
for task in ['position','single','multi','language']:
 m=Model(task,0,'transformer');x=jnp.array(np.random.default_rng(0).integers(0,65,(4,32 if task=='language' else 16)))
 pos=[10] if task=='single' else [10,12,14] if task=='multi' else [x.shape[1]-1]
 def ref(mod):
  y=mod.base(x,full=True)[:,pos]
  return y if task in ['single','multi'] else y[:,0]
 np.testing.assert_allclose(m(x),ref(m),rtol=1e-4,atol=1e-5)
 g1=nnx.grad(lambda mod:jnp.square(mod(x)).mean())(m)
 g2=nnx.grad(lambda mod:jnp.square(ref(mod)).mean())(m)
 for a,b in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)):np.testing.assert_allclose(a,b,rtol=1e-3,atol=1e-5)
 print('optimized logits/gradients passed',task)
for arch in ['ours','transformer']:
 m=Model('position',0,arch);ordinary=nnx.clone(m);tx=optax.adam(.001)
 opt=nnx.Optimizer(ordinary,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,8,16)));y=jnp.array(rng.integers(0,16,(5,8)))
 state,run=compiled_training(m,tx);state,loss=run(state,x,y)
 for xx,yy in zip(x,y):learn(ordinary,opt,xx,yy)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((ordinary,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('compiled full-state parity passed',arch)
