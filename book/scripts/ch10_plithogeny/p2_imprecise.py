# p2_imprecise.py -- checks for Section 12 (imprecise probability, imprecise statistics and credal sets)
import numpy as np, itertools, json
from scipy.optimize import linprog
out={}
# (1) credal set = hull of two expert distributions (not event-determined)
P1=np.array([0.2,0.3,0.5]); P2=np.array([0.4,0.5,0.1])
events=[e for r in (1,2) for e in itertools.combinations(range(3),r)]
low={e:min(P1[list(e)].sum(),P2[list(e)].sum()) for e in events}
out['event_lowers']={str(e):round(v,4) for e,v in low.items()}
A_ub=[];b_ub=[]
for e,v in low.items():
    A_ub.append([-1.0 if i in e else 0 for i in range(3)]); b_ub.append(-v)
def core_ext(f,sense=1):
    r=linprog(sense*np.array(f),A_ub=A_ub,b_ub=b_ub,A_eq=[[1,1,1]],b_eq=[1],bounds=(0,1),method='highs'); return sense*r.fun
# vertices of core: enumerate by brute force
verts=[]
for idx in itertools.combinations(range(len(A_ub)),2):
    A=np.array([A_ub[i] for i in idx]+[[1,1,1]]); b=np.array([b_ub[i] for i in idx]+[1])
    try: x=np.linalg.solve(A,b)
    except: continue
    if (np.array(A_ub)@x<=np.array(b_ub)+1e-9).all() and (x>=-1e-9).all():
        if not any(np.allclose(x,v) for v in verts): verts.append(x)
out['core_vertices']=[list(np.round(v,4)) for v in verts]
loss=np.array([0,100,10])  # payoff of the production plan in Section 12.3
out['hull_E']=[round(min(P1@loss,P2@loss),3),round(max(P1@loss,P2@loss),3)]
out['core_E']=[round(core_ext(loss,1),3),round(core_ext(loss,-1),3)]
# (2) Dempster vs generalized Bayes conditioning
foc={('a',):0.3,('b','c'):0.4,('a','b','c'):0.3}
B={'a','b'}
# Dempster: intersect with B, normalise by Pl(B)
mB={}
for f,w in foc.items():
    g=tuple(sorted(set(f)&B))
    if g: mB[g]=mB.get(g,0)+w
Z=sum(mB.values()); mB={k:v/Z for k,v in mB.items()}
bel=sum(v for k,v in mB.items() if set(k)<={'a'}); pl=sum(v for k,v in mB.items() if 'a' in k)
out['dempster_a_given_B']=[round(bel,4),round(pl,4)]
# generalized Bayes: vertices of credal set = allocations of each focal mass to an element
vals=[]
lists=[list(f) for f in foc]
for choice in itertools.product(*lists):
    p={'a':0,'b':0,'c':0}
    for f,x in zip(foc,choice): p[x]+=foc[f]
    if p['a']+p['b']>0: vals.append(p['a']/(p['a']+p['b']))
out['GBR_a_given_B']=[round(min(vals),4),round(max(vals),4)]
# (3) credal classifier with IDM, s=2
for n in [(9,7,2),(18,14,4)]:
    n=np.array(n);N=n.sum();s=2
    lo=n/(N+s); hi=(n+s)/(N+s)
    undominated=[k for k in range(3) if not any(lo[j]>hi[k] for j in range(3) if j!=k)]
    out[f'ncc_{list(n)}']=dict(intervals=[[round(a,3),round(b,3)] for a,b in zip(lo,hi)],undominated=[k+1 for k in undominated],
       triples=[[round(a,3),round(b-a,3),round(1-b,3)] for a,b in zip(lo,hi)])
# (4) Manski bounds vs plithogenic IDM (North region of Table 7)
ns,nn,nq=112,38,20;N=ns+nn+nq
out['manski']=[round(ns/N,4),round((ns+nq)/N,4)]
for s in [2,1,0.1]: out[f'pidm_s{s}']=[round(ns/(N+s),4),round((ns+nq+s)/(N+s),4)]
assert out['hull_E']==[35.0,51.0] and out['core_E']==[33.0,53.0]
assert out['dempster_a_given_B']==[0.3,0.6] and out['GBR_a_given_B']==[0.3,1.0]
json.dump(out,open('p2_imprecise.json','w'),indent=1)
print(json.dumps(out,indent=1)); print('ALL CHECKS PASSED')
