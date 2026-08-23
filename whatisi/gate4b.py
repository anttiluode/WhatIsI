from __future__ import annotations
from dataclasses import dataclass
import random, json, math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Config:
    n_agents:int=4; pool:int=8; steps:int=32
    silent_windows:tuple=((7,10),(19,22),(28,30))
    surface_swap_step:int=12; transfer_step:int=23
    d_model:int=48; n_heads:int=4; n_layers:int=1; ff:int=96; memory_dim:int=20
    train_lives:int=512; test_lives:int=128; batch_size:int=64; epochs:int=9; lr:float=3e-3
    actor_weight:float=1.0; pron_weight:float=0.8

ACTIONS=("left","right","say_red","say_blue","say_green","wait"); WAIT=5
POS=(-1,0,1)
def silent(t,c): return any(a<=t<b for a,b in c.silent_windows)

def gen_life(seed,c:Config):
    r=np.random.default_rng(seed)
    shapes=r.choice(c.pool,c.n_agents,False); names=r.choice(c.pool,c.n_agents,False); voices=r.choice(c.pool,c.n_agents,False)
    pos=r.integers(-1,2,c.n_agents); ctl=int(r.integers(c.n_agents))
    rows={k:[] for k in ['shape','name','voice','pos','motor','sp_name','ad_name','pron','actor_name','pron_pos','fb_shape','fb_name','fb_voice','fb_pos','active','self_shape']}
    for t in range(c.steps):
        if t==c.surface_swap_step:
            names=names[r.permutation(c.n_agents)]; voices=voices[r.permutation(c.n_agents)]
        if t==c.transfer_step:
            ctl=int(r.choice([i for i in range(c.n_agents) if i!=ctl]))
        order=r.permutation(c.n_agents)
        rows['shape'].append(shapes[order].copy()); rows['name'].append(names[order].copy()); rows['voice'].append(voices[order].copy()); rows['pos'].append(pos[order].copy()+1)
        active=not silent(t,c); act=int(r.integers(5)) if active else WAIT
        sp=int(r.integers(c.n_agents)); ad=int(r.choice([i for i in range(c.n_agents) if i!=sp])); pr=int(r.integers(2)); ref=sp if pr==0 else ad
        rows['motor'].append(act); rows['sp_name'].append(int(names[sp])); rows['ad_name'].append(int(names[ad])); rows['pron'].append(pr)
        rows['actor_name'].append(int(names[ctl]) if active else -100); rows['pron_pos'].append(int(pos[ref]+1)); rows['active'].append(active); rows['self_shape'].append(int(shapes[ctl]))
        if active:
            if act==0: pos[ctl]=max(-1,pos[ctl]-1)
            elif act==1: pos[ctl]=min(1,pos[ctl]+1)
            rows['fb_shape'].append(int(shapes[ctl])); rows['fb_name'].append(int(names[ctl])); rows['fb_voice'].append(int(voices[ctl])); rows['fb_pos'].append(int(pos[ctl]+1))
        else:
            rows['fb_shape'].append(0); rows['fb_name'].append(0); rows['fb_voice'].append(0); rows['fb_pos'].append(1)
        dj=int(r.choice([i for i in range(c.n_agents) if i!=ctl])); da=int(r.integers(2)); pos[dj]=max(-1,pos[dj]-1) if da==0 else min(1,pos[dj]+1)
    return {k:np.asarray(v) for k,v in rows.items()}

def make(seed,n,c):
    ls=[gen_life(seed+i,c) for i in range(n)]; return {k:np.stack([x[k] for x in ls]) for k in ls[0]}

class Model(nn.Module):
    def __init__(self,c:Config):
        super().__init__(); self.c=c; d=c.d_model
        self.shape=nn.Embedding(c.pool,d); self.name=nn.Embedding(c.pool,d); self.voice=nn.Embedding(c.pool,d); self.pos=nn.Embedding(3,d)
        self.motor=nn.Embedding(len(ACTIONS),d); self.pron=nn.Embedding(2,d); self.kind=nn.Embedding(3,d); self.sp_proj=nn.Linear(d,d,bias=False); self.ad_proj=nn.Linear(d,d,bias=False)
        layer=nn.TransformerEncoderLayer(d,c.n_heads,c.ff,0.05,activation='gelu',batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,c.n_layers); self.norm=nn.LayerNorm(d)
        self.memq=nn.Linear(c.memory_dim,d); self.key=nn.Linear(d,d)
        self.pron_head=nn.Linear(d,3)
        self.fb=nn.Linear(d*5,d); self.mem_update=nn.GRUCell(d,c.memory_dim)
    def scene(self,shape,name,voice,pos,motor,sp,ad,pron):
        ag=self.shape(shape)+self.name(name)+self.voice(voice)+self.pos(pos)+self.kind.weight[0]
        mot=self.motor(motor)+self.kind.weight[1]
        q=self.sp_proj(self.name(sp))+self.ad_proj(self.name(ad))+self.pron(pron)+self.kind.weight[2]
        x=torch.cat([ag,mot[:,None,:],q[:,None,:]],1); h=self.norm(self.enc(x))
        return h[:,:self.c.n_agents], h[:,-1]
    def heads(self,agent_h,q_h,names,mem):
        scores=torch.einsum('bd,bad->ba',self.memq(mem),self.key(agent_h))/math.sqrt(self.c.d_model)
        logits=torch.full((len(mem),self.c.pool),-1e9,device=mem.device); logits.scatter_(1,names,scores)
        return logits,self.pron_head(q_h)
    def update(self,fb_shape,fb_name,fb_voice,fb_pos,motor,mem,active):
        x=torch.cat([self.shape(fb_shape),self.name(fb_name),self.voice(fb_voice),self.pos(fb_pos),self.motor(motor)],-1)
        cand=self.mem_update(torch.tanh(self.fb(x)),mem)
        return torch.where(active[:,None],cand,mem)

def batch_to_torch(b,idx,dev): return {k:torch.as_tensor(v[idx],device=dev) for k,v in b.items()}

def train(seed=1,c=None,device=None):
    c=c or Config(); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.set_num_threads(1); dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    m=Model(c).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=c.lr,weight_decay=1e-4); data=make(seed*10000,c.train_lives,c)
    for ep in range(c.epochs):
        order=np.random.permutation(c.train_lives)
        for lo in range(0,c.train_lives,c.batch_size):
            idx=order[lo:lo+c.batch_size]; b=batch_to_torch(data,idx,dev); B=len(idx); mem=torch.zeros(B,c.memory_dim,device=dev); loss=0.; denom=0.
            for t in range(c.steps):
                ah,qh=m.scene(b['shape'][:,t],b['name'][:,t],b['voice'][:,t],b['pos'][:,t],b['motor'][:,t],b['sp_name'][:,t],b['ad_name'][:,t],b['pron'][:,t])
                al,pl=m.heads(ah,qh,b['name'][:,t],mem); act=b['active'][:,t].bool()
                if act.any(): loss=loss+c.actor_weight*F.cross_entropy(al[act],b['actor_name'][act,t]); denom+=c.actor_weight
                loss=loss+c.pron_weight*F.cross_entropy(pl,b['pron_pos'][:,t]); denom+=c.pron_weight
                mem=m.update(b['fb_shape'][:,t],b['fb_name'][:,t],b['fb_voice'][:,t],b['fb_pos'][:,t],b['motor'][:,t],mem,act)
            loss=loss/denom; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    return m,c

@torch.no_grad()
def rollout(m,c,seed,n,zero=False,collect=False):
    dev=next(m.parameters()).device; data=make(seed,n,c); b={k:torch.as_tensor(v,device=dev) for k,v in data.items()}; mem=torch.zeros(n,c.memory_dim,device=dev)
    aps=[]; pps=[]; hs=[]
    for t in range(c.steps):
        use=torch.zeros_like(mem) if zero else mem; hs.append(use.clone())
        ah,qh=m.scene(b['shape'][:,t],b['name'][:,t],b['voice'][:,t],b['pos'][:,t],b['motor'][:,t],b['sp_name'][:,t],b['ad_name'][:,t],b['pron'][:,t]); al,pl=m.heads(ah,qh,b['name'][:,t],use)
        aps.append(al.argmax(1)); pps.append(pl.argmax(1)); act=b['active'][:,t].bool(); mem=m.update(b['fb_shape'][:,t],b['fb_name'][:,t],b['fb_voice'][:,t],b['fb_pos'][:,t],b['motor'][:,t],use,act)
    ap=torch.stack(aps,1); pp=torch.stack(pps,1); active=b['active'].bool(); tg=torch.arange(c.steps,device=dev)[None,:]
    out={'actor_accuracy':float((ap[active]==b['actor_name'][active]).float().mean()),'pronoun_accuracy':float((pp==b['pron_pos']).float().mean())}
    sw=(tg>=c.surface_swap_step)&(tg<c.transfer_step)&active; tr=(tg>c.transfer_step)&active
    out['post_surface_swap_actor_accuracy']=float((ap[sw]==b['actor_name'][sw]).float().mean()); out['post_transfer_actor_accuracy']=float((ap[tr]==b['actor_name'][tr]).float().mean())
    if collect: out['hidden']=torch.stack(hs,1).cpu().numpy(); out['data']=data
    return out

def ridge(H,y,K):
    X=np.c_[H,np.ones(len(H))]; return np.linalg.pinv(X)@np.eye(K)[y]
def racc(H,y,W): return float(np.mean(np.argmax(np.c_[H,np.ones(len(H))]@W,1)==y))
class Head(nn.Module):
    """Held-out deictic selector. No owner labels: it only sees task targets."""
    def __init__(self,d,K): super().__init__(); self.sel=nn.Linear(d,K)
    def forward(self,m,q):
        w=torch.softmax(self.sel(m),1)
        return torch.sum(w*q,1)

def analyze(m,c,seed=50000):
    dev=next(m.parameters()).device; r=rollout(m,c,seed,c.test_lives,collect=True); H=r.pop('hidden'); d=r.pop('data'); tg=np.arange(c.steps)[None,:]
    valid=(~d['active'])&(tg>=10); half=c.test_lives//2; tr=valid[:half]; te=valid[half:]; Htr=H[:half][tr]; ytr=d['self_shape'][:half][tr]; Hte=H[half:][te]; yte=d['self_shape'][half:][te]
    W=ridge(Htr,ytr,c.pool); r['linear_probe_self_shape']=racc(Hte,yte,W)
    rng=np.random.default_rng(seed+9); qtr=rng.normal(size=(len(Htr),c.pool)).astype('float32'); yv=qtr[np.arange(len(Htr)),ytr]; qte=rng.normal(size=(len(Hte),c.pool)).astype('float32'); yteval=qte[np.arange(len(Hte)),yte]
    take=min(256,len(Htr)); ix=rng.choice(len(Htr),take,False); head=Head(c.memory_dim,c.pool).to(dev); o=torch.optim.AdamW(head.parameters(),lr=7e-3); mt=torch.from_numpy(Htr[ix]).float().to(dev); qt=torch.from_numpy(qtr[ix]).to(dev); yt=torch.from_numpy(yv[ix]).to(dev)
    for _ in range(300):
        p=head(mt,qt); l=F.mse_loss(p,yt); o.zero_grad(); l.backward(); o.step()
    with torch.no_grad(): p=head(torch.from_numpy(Hte).float().to(dev),torch.from_numpy(qte).to(dev)).cpu().numpy()
    var=max(np.var(yteval),1e-9); r['heldout_join_nmse']=float(np.mean((p-yteval)**2)/var)
    base=Head(c.memory_dim,c.pool).to(dev); o2=torch.optim.AdamW(base.parameters(),lr=7e-3); z=torch.zeros_like(mt)
    for _ in range(300):
        p2=base(z,qt); l=F.mse_loss(p2,yt); o2.zero_grad(); l.backward(); o2.step()
    with torch.no_grad(): pb=base(torch.zeros(len(Hte),c.memory_dim,device=dev),torch.from_numpy(qte).to(dev)).cpu().numpy()
    r['zero_memory_join_nmse']=float(np.mean((pb-yteval)**2)/var)
    cents=np.stack([Htr[ytr==s].mean(0) if np.any(ytr==s) else np.zeros(c.memory_dim) for s in range(c.pool)]).astype('float32'); cand=[]
    for i,trues in enumerate(yte):
        opts=[s for s in range(c.pool) if s!=trues and np.any(ytr==s)]
        if opts: cand.append((i,int(trues),int(rng.choice(opts))))
    mi=np.stack([cents[cf] for i,tru,cf in cand]); qi=np.stack([qte[i] for i,tru,cf in cand])
    with torch.no_grad(): pi=head(torch.from_numpy(mi).to(dev),torch.from_numpy(qi).to(dev)).cpu().numpy()
    r['counterfactual_intervention_rate']=float(np.mean([abs(v-qte[i,cf])<abs(v-qte[i,tru]) for v,(i,tru,cf) in zip(pi,cand)]))
    zroll=rollout(m,c,seed+1000,c.test_lives,zero=True); r['zero_memory_actor_accuracy']=zroll['actor_accuracy']
    r['checks']={'actor_requires_state':r['actor_accuracy']>.70 and r['zero_memory_actor_accuracy']<.40,'pronouns_role_resolved':r['pronoun_accuracy']>.85,'self_decodable_silence':r['linear_probe_self_shape']>.75,'new_task_reuses_state':r['heldout_join_nmse']<.35 and r['zero_memory_join_nmse']>.75,'intervention_moves_referent':r['counterfactual_intervention_rate']>.70}; r['pass']=all(r['checks'].values()); return r

def run(seed=3,c=None,device=None):
    m,c=train(seed,c,device); return analyze(m,c,seed+50000)

def run_many(seeds=(3,4,5),c=None,device=None):
    rows=[]
    for seed in seeds:
        rows.append({'seed':int(seed), **run(int(seed), c, device)})
    keys=['actor_accuracy','zero_memory_actor_accuracy','post_surface_swap_actor_accuracy','post_transfer_actor_accuracy','pronoun_accuracy','linear_probe_self_shape','heldout_join_nmse','zero_memory_join_nmse','counterfactual_intervention_rate']
    summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
    summary['rows']=rows; summary['pass']=bool(all(r['pass'] for r in rows)); return summary

if __name__=='__main__': print(json.dumps(run_many(),indent=2,sort_keys=True))
