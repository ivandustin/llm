from transformer_blocks_test import *
from speed_benchmark import compiled_training
from readout_experiments import learn
import optax
for arch in DEPTHS:
 for task in ['position','single','multi','language']:
  m=Model(task,0,arch);x=jnp.array(np.random.default_rng(0).integers(0,65,(2,32 if task=='language' else 16)))
  pos=[10] if task=='single' else [10,12,14] if task=='multi' else [x.shape[1]-1]
  def ref(mod):
   base=mod.base;h=base.token(x)+base.position(jnp.arange(x.shape[1]))
   for block in base.blocks:h=block(h)
   logits=base.head(base.norm(h[:,pos]))
   return logits if task in ['single','multi'] else logits[:,0]
  np.testing.assert_allclose(m(x),ref(m),rtol=1e-4,atol=1e-5)
  g1=nnx.grad(lambda mod:jnp.square(mod(x)).mean())(m);g2=nnx.grad(lambda mod:jnp.square(ref(mod)).mean())(m)
  for a,b in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)):np.testing.assert_allclose(a,b,rtol=1e-3,atol=1e-5)
  print('logits/gradient parity',arch,task,'params',count(m),'FF',dimensions(task,DEPTHS[arch])[2],flush=True)
for arch in DEPTHS:
 m=Model('position',0,arch);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,4,16)));y=jnp.array(rng.integers(0,16,(5,4)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step full-state parity',arch,flush=True)
