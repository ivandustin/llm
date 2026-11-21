"""Check matched parameters, raw-pool sigmoid formula, and compiled optimizer parity."""
from benchmark import nnx,np,jax,jnp,count
from parallel_mixture_100k import Model as Control
from sigmoid_no_norm import Model
from readout_experiments import learn
from speed_benchmark import compiled_training
import optax
for task in ['position','language']:
    m=Model(task,0);control=Control(task,0,'n1')
    assert count(m)==count(control)==(81153 if task=='language' else 100000)
    a=dict(nnx.to_flat_state(nnx.state(m,nnx.Param)));b=dict(nnx.to_flat_state(nnx.state(control,nnx.Param)))
    assert a.keys()==b.keys()
    for key,value in a.items():np.testing.assert_array_equal(value,b[key])
    x=jnp.array(np.random.default_rng(0).integers(0,31,(4,32 if task=='language' else 16)))
    z=(m.base.token(x)*m.base.position(jnp.arange(x.shape[1]))).sum(1)
    head=m.base.branches[0]
    expected=head.down(1/(1+jnp.exp(-head.up(z))))
    np.testing.assert_allclose(m(x),expected,rtol=1e-5,atol=1e-6)
    other=nnx.clone(m);tx=optax.adam(.001);optimizer=nnx.Optimizer(other,tx,wrt=nnx.Param)
    rng=np.random.default_rng(7);xs=jnp.array(rng.integers(0,31,(5,4,x.shape[1])));ys=jnp.array(rng.integers(0,16,(5,4)))
    state,run=compiled_training(m,tx);state,losses=run(state,xs,ys)
    manual=[learn(other,optimizer,xx,yy) for xx,yy in zip(xs,ys)]
    np.testing.assert_allclose(losses,manual,rtol=1e-4,atol=1e-5)
    for aa,bb in zip(jax.tree.leaves(state),jax.tree.leaves(nnx.state((other,optimizer)))):np.testing.assert_allclose(aa,bb,rtol=1e-4,atol=1e-5)
    print('same initial weights/count, unnormalized sigmoid output, five-step full-state parity:',task,flush=True)
