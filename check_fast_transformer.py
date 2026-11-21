from benchmark_100k import Model
from benchmark import nnx,jnp,np
from capabilities import generate
from mqar_test import generate as assoc
import optax
for task in ['position','single','multi','language']:
 m=Model(task,0,'transformer')
 if task=='language':x=np.random.default_rng(2).integers(0,65,(4,32),dtype=np.int32);y=np.arange(4,dtype=np.int32)
 elif task=='position':x,y,_=generate('indexed_copy',16,4,np.random.default_rng(2))
 else:
  x,y=assoc(4,np.random.default_rng(2))
  if task=='single':y=y[:,:1]
 def old(m,x):
  return jnp.stack([m.base(x[:,:n]) for n in ([11] if task=='single' else [11,13,15])],1) if task in ['single','multi'] else m.base(x)
 np.testing.assert_allclose(m(x),old(m,x),rtol=1e-4,atol=1e-5)
 def loss(m,fast):return optax.softmax_cross_entropy_with_integer_labels(m(x) if fast else old(m,x),y).mean()
 a=nnx.grad(lambda m:loss(m,True))(m);b=nnx.grad(lambda m:loss(m,False))(m)
 from benchmark import jax
 for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(u,v,rtol=1e-3,atol=1e-5)
 print(task,'logits and gradients match',flush=True)
