"""Convert MATPOWER case33bw.m to project CSV files and run a radial PF check."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/case33/case33bw.m"
OUT = ROOT / "data/case33"

def matrix(text, name):
    m = re.search(rf"mpc\.{name}\s*=\s*\[([\s\S]*?)\];", text)
    if not m: raise ValueError(name)
    rows=[]
    for line in m.group(1).splitlines():
        line=line.split('%',1)[0].strip()
        if not line: continue
        line=line.rstrip(';').strip()
        if line:
            rows.append([float(x) for x in re.split(r'\s+', line)])
    return np.asarray(rows)

def main():
    text=SRC.read_text()
    bus=matrix(text,'bus'); branch=matrix(text,'branch')
    # MATPOWER's case33bw includes five normally-open tie branches after the
    # original 32 radial branches; the base topology uses the radial part.
    branch=branch[:32]
    nodes=pd.DataFrame({'node':bus[:,0].astype(int), 'type':bus[:,1].astype(int),
        'pd_kw':bus[:,2], 'qd_kvar':bus[:,3], 'vm_init_pu':bus[:,7]})
    lines=pd.DataFrame({'line':np.arange(1,len(branch)+1), 'from_node':branch[:,0].astype(int),
        'to_node':branch[:,1].astype(int), 'r_ohm':branch[:,2], 'x_ohm':branch[:,3],
        'rate_a':branch[:,5], 'status':branch[:,10].astype(int)})
    nodes.to_csv(OUT/'nodes.csv',index=False); lines.to_csv(OUT/'lines.csv',index=False)
    # Backward-forward sweep on the original radial network (base 12.66 kV, 10 MVA).
    zbase=(12.66**2)/10.0; r=branch[:,2]/zbase; x=branch[:,3]/zbase
    p=bus[:,2]/10000; q=bus[:,3]/10000; v=np.ones(len(bus),complex); parent={int(b):int(a) for a,b in branch[:,:2]}
    children={i:[] for i in range(1,34)}
    for a,b in parent.items(): children[a].append(b)
    order=list(range(33,1,-1));
    for _ in range(100):
        s=(p+1j*q)/np.conj(v); flow=np.zeros(32,complex)
        # case33bw lists branches parent-to-child; aggregate from leaves upward.
        for k in range(31, -1, -1):
            a,b=branch[k,:2].astype(int)
            flow[k]=s[b-1]+sum(flow[j] for j,(aa,bb) in enumerate(branch[:,:2].astype(int)) if aa==b)
        vn=np.ones(33,complex)
        for k,(a,b) in enumerate(branch[:,:2].astype(int)):
            vn[b-1]=vn[a-1]-(r[k]+1j*x[k])*flow[k]/np.conj(v[a-1])
        if np.max(np.abs(vn-v))<1e-10: v=vn; break
        v=vn
    loss=sum((abs(flow[k])**2)*(r[k]+1j*x[k]) for k in range(32))
    summary={'bus_count':33,'branch_count':32,'total_load_kw':float(nodes.pd_kw.sum()),
             'min_voltage_pu':float(np.min(np.abs(v))), 'max_voltage_pu':float(np.max(np.abs(v))),
             'estimated_loss_kw':float(loss.real*10000)}
    pd.DataFrame([summary]).to_json(OUT/'baseline_summary.json',orient='records',indent=2)
    print(summary)

if __name__=='__main__': main()
