import json
from experiments.benchmark import jax,jnp,np
from experiments.longer_training_100k import TokenProductModel
from experiments.validate_layer_depth import load_weights
from experiments.mqar_test import generate
x,y=generate(4096,np.random.default_rng(2026112501)); x=x[:,:11]; y=y[:,0]
rows=[]
for seed in range(3):
 m=TokenProductModel('single',seed,variant='token_product_raw_gelu_add',width=256,hidden=201)
 load_weights(m,f'experiments/raw-gelu-add-d256-30k/d256-n1-seed-{seed}/single-step-30000.npz')
 b=m.base.base; p=b.embeds.embeds[0](jnp.arange(10)); t=b.embeds.embeds[1](jnp.array(x[:,:10])); q=b.embeds.embeds[1](jnp.array(x[:,10])); v=p[None,:,:]*t*q[:,None,:]; raw=v; gel=jax.nn.gelu(v)
 def read(z):
  z=(z-z.mean(-1,keepdims=True))/jnp.sqrt(z.var(-1,keepdims=True)+1e-5)
  return b.fn.b(jax.nn.gelu(b.fn.a(z)))
 def margin(z):
  logits=read(z); target=jnp.take_along_axis(logits,jnp.array(y)[:,None],axis=1)[:,0]; other=jnp.max(jnp.where(jax.nn.one_hot(jnp.array(y),96,dtype=bool),-1e9,logits),axis=1); return target-other
 total=(raw+gel).sum(1); baseline=margin(total); pred=np.asarray(read(total)).argmax(-1); assert np.all(pred==np.asarray(m(jnp.array(x))).argmax(-1)[:,0])
 match=(x[:,:10:2]==x[:,10,None]).argmax(1); idx=np.arange(len(x)); out={'seed':seed,'accuracy':float((pred==y).mean()),'margin':float(baseline.mean())}
 for name,branch in [('raw',raw),('gelu',gel),('both',raw+gel)]:
  for category,positions in [('matching_key',2*match),('matching_value',2*match+1)]:
   out[name+'_'+category]=float((baseline-margin(total-branch[idx,positions])).mean())
  for category,offset in [('other_keys',0),('other_values',1)]:
   drops=[]
   for k in range(5):
    d=np.asarray(baseline-margin(total-branch[:,2*k+offset])); drops.append(d)
   drops=np.stack(drops,1); out[name+'_'+category]=float(drops[np.arange(5)[None,:]!=match[:,None]].mean())
 rows.append(out)
print(json.dumps(rows,indent=2))
