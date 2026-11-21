from residual_depth_100k_test import *
from speed_benchmark import compiled_training
from readout_experiments import learn
import optax
expected=dict(six=(99184,94473),eight=(98831,94647),twelve=(99328,95733))
for arch in SPECS:
 for task in ['position','language']:
  a=Model(task,0,arch,False);b=Model(task,0,arch,True)
  assert count(a)==count(b)==expected[arch][task=='language']
  sa=dict(nnx.to_flat_state(nnx.state(a,nnx.Param)));sb=dict(nnx.to_flat_state(nnx.state(b,nnx.Param)))
  assert sa.keys()==sb.keys()
  for k,v in sa.items():np.testing.assert_array_equal(v,sb[k])
  print('same initialization and parameter count',arch,task,count(b),flush=True)
 m=Model('position',0,arch);other=nnx.clone(m);tx=optax.adam(.001);opt=nnx.Optimizer(other,tx,wrt=nnx.Param)
 rng=np.random.default_rng(7);x=jnp.array(rng.integers(0,31,(5,4,16)));y=jnp.array(rng.integers(0,16,(5,4)))
 state,run=compiled_training(m,tx);state,losses=run(state,x,y)
 manual=[learn(other,opt,xx,yy) for xx,yy in zip(x,y)]
 np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
 for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,opt)))):np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
 print('five-step complete-state parity',arch,flush=True)
