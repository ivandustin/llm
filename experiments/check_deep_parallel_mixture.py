from experiments.deep_parallel_mixture_100k import *
from experiments.speed_benchmark import compiled_training
from experiments.readout_experiments import learn
import optax
for arch,n in BRANCHES.items():
 for task in ['position','language']:
  m=Model(task,0,arch);vocab=65 if task=='language' else 96;context=32 if task=='language' else 16;h=hidden_for_branches(n)
  expected=(context+vocab)*64+n*((64+1)*h+5*(h*h+h)+(h+1)*vocab)
  assert count(m)==expected and count(m)<=100000
  x=jnp.array(np.random.default_rng(0).integers(0,31,(4,context)))
  base=m.base;z=(base.token(x)*base.position(jnp.arange(context))).sum(1)
  z=(z-z.mean(-1,keepdims=True))/jnp.sqrt(z.var(-1,keepdims=True)+1e-5)
  probs=jnp.stack([jax.nn.softmax(branch(z)) for branch in base.branches]).mean(0)
  np.testing.assert_allclose(jnp.exp(m(x)),probs,rtol=1e-5,atol=1e-7)
  np.testing.assert_allclose(jnp.exp(m(x)).sum(-1),1,rtol=1e-5,atol=1e-6)
  for grad in jax.tree.leaves(nnx.grad(lambda mod:-mod(x)[:,0].mean())(m)):assert np.isfinite(grad).all()
  print('probability mean/count/gradient checks',arch,task,h,count(m),flush=True)
 m=Model('position',0,arch);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,4,16)));y=jnp.array(rng.integers(0,16,(5,4)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step complete-state parity',arch,flush=True)
