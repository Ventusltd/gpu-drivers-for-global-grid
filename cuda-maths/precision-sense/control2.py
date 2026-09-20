import sys, json, numpy as np, cupy as cp
sys.path.insert(0, r"(local path removed)")
from solar_core import *
import cpu_vec
from cpu_vec import build_tables, kernel
from cupyx import scatter_max
AV = axis_values()
SH = dict(AV); SH["L"]=[v+8.125 for v in AV["L"]]; SH["Tcond"]=[v+5.0 for v in AV["Tcond"]]
SH["Tmin"]=[v+2.5 for v in AV["Tmin"]]; SH["Thot"]=[v+2.5 for v in AV["Thot"]]
SH["bif"]=[v+0.025 for v in AV["bif"]]; SH["Rc"]=[0.000375,0.00075,0.00125]
cells=np.arange(N_CELLS); dTn_=(cells//120)%7; dc_=(cells//840)%3
Voc0=np.array([c[1] for c in CLASSES])[dc_]; beta0=np.array([c[5] for c in CLASSES])[dc_]
out={}
for tag,av in (("base",AV),("shifted",SH)):
    cpu_vec._AV=av; T=build_tables(cp)
    best=cp.zeros(N_CELLS,dtype=cp.int64); done=0
    while done<SPACE:
        n=min(1<<22,SPACE-done); idx=cp.arange(done,done+n,dtype=cp.int64)
        _,pk,cell,_=kernel(cp,T,idx); g=pk>0
        if bool(g.any()): scatter_max(best,cell[g],pk[g])
        done+=n
    best=cp.asnumpy(best); bN=(best>>44)&0x3F; hv=best>0
    Tm=np.array(av["Tmin"])[dTn_]
    r={}
    for rd,allow in (("A_+10K",T_ALLOW_K),("B_no_allowance",0.0)):
        Nmax=np.floor(1500.0/(Voc0*(1.0+beta0/100.0*((Tm+allow)-25.0))))
        pred=np.minimum(34.0,Nmax); pred=np.where(pred<20,0,pred)
        ok=(bN==pred)&hv
        exc=np.where(hv&~ok)[0][:10]
        r[rd]=dict(share=float(ok.sum()/hv.sum()),ok=int(ok.sum()),exceptions=[int(e) for e in exc])
    out[tag]=dict(cells=int(hv.sum()),by_reading=r)
print(json.dumps(out))
cp.get_default_memory_pool().free_all_blocks()
