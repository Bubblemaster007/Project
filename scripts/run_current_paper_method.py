"""Reuse the current Code_Paper SC-RCRB and certified calendar methods unchanged."""
from pathlib import Path
import argparse, sys, json, hashlib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--paper-root',default='/Users/bubble/Code_Paper/ExtremeScene')
    parser.add_argument('--dataset',default='singleton')
    parser.add_argument('--year',type=int,default=2021)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--regenerate',action='store_true')
    args=parser.parse_args(); paper=Path(args.paper_root);sys.path.insert(0,str(paper))
    from annual_historical_calendar import fit_calendar,sample_calendar,placement_candidates
    from annual_transition_certified import repair_certified
    from annual_calendar_assignment import assign
    root=paper/'publication_protocol';data=root/'data/main'/args.dataset
    out=PROJECT/'outputs/current_paper'/args.dataset/str(args.year)/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    fitted=json.loads((data/'fitted_parameters.json').read_text())
    raw=pd.read_csv(fitted['source'],parse_dates=['time']).sort_values('time')
    train=raw[raw.time<pd.Timestamp(fitted['train_end'])];target=raw[raw.time.dt.year.eq(args.year)]
    if target.empty:raise ValueError('Target year unavailable')
    meta=pd.read_csv(data/'meta_train.csv');mask=np.load(data/'event_mask_train.npy').astype(bool)
    bank=root/'final_candidate/downstream'/args.dataset
    cond=pd.read_csv(bank/'planning_conditions.csv')
    assert list(cond.sample_id)==list(meta.sample_id)
    if args.regenerate:
        from publication_structure_calibrated import generate_conditions
        selected=pd.read_csv(root/'final_candidate/selected_main_seed123.csv').set_index('dataset').loc[args.dataset]
        hours=(cond.start_hour.to_numpy()[:,None]+np.arange(36))%24
        generated,audit=generate_conditions(args.dataset,123,'test',cond,mask,hours,selected.bridge,5123,float(selected.corr_weight),float(selected.acf_weight))
        np.save(out/'planning.npy',generated);audit.to_csv(out/'planning_selection_audit.csv',index=False)
    else:generated=np.load(bank/'planning.npy')
    cond.to_csv(out/'planning_conditions.csv',index=False)
    # A common positive multiplier preserves the paper's net-load identities.
    tr=train[['load','wind_power','solar_power']].to_numpy(float)
    factor=3715.0/tr[:,0].max();tr*=factor
    b=target[['load','wind_power','solar_power']].to_numpy(float)*factor;events=generated.transpose(0,2,1)*factor
    times=pd.DatetimeIndex(target.time);net=np.array([1.,-1.,-1.])
    scale=np.maximum(np.quantile(tr,.95,axis=0),1e-6);upper=tr.max(axis=0)
    diff=np.diff(tr,axis=0)[train.time.diff().eq(pd.Timedelta(hours=1)).to_numpy()[1:]]
    ramp=np.maximum(np.quantile(abs(diff),.99,axis=0),1e-6);nr=max(float(np.quantile(abs(diff@net),.99)),1e-6)
    dark_hours=[h for h in range(24) if np.quantile(tr[train.time.dt.hour.eq(h),2],.99)<.01*upper[2]]
    dark=np.isin(times.hour,dark_hours);tau=np.array([fitted['tau_by_month'][str(m)] for m in times.month])*factor
    calendar=fit_calendar(meta,train.time);requests,templates=sample_calendar(calendar,meta,times,args.seed)
    requests.to_csv(out/'requests.csv',index=False);templates.to_csv(out/'templates.csv',index=False)
    candidates=[];cache={};failures=[]
    for i,req in enumerate(requests.to_dict('records')):
        eid=req['bank_index'];x=np.clip(events[eid],0,upper);x[np.isin((req['start_hour']+np.arange(36))%24,dark_hours),2]=0
        for s in placement_candidates(times,req['desired_start_time'],max_shift_days=7):
            shift=times[s]-pd.Timestamp(req['desired_start_time']);core=pd.Timestamp(req['desired_core_time'])+shift
            if core.month!=req['core_month']:continue
            result=repair_certified(b,x,s,widths=[0,2,4,8],scale=scale,ramp=ramp,net_ramp=nr,upper=upper,tau=tau,dark=dark,core_mask=mask[eid])
            if not result.success:failures.append(dict(request_index=i,start=s,reason=result.reason));continue
            h=result.width;j=len(candidates);cache[j]=(result.annual[s-h:s+36+h].copy(),x,result.realized_added_risk)
            candidates.append(dict(request_index=i,bank_index=eid,start=s,left=s-h-1,right=s+36+h+1,width=h,shift_days=shift.total_seconds()/86400))
        print(f'certified request {i+1}/{len(requests)}',flush=True)
    chosen,proof=assign(candidates,requests.to_dict('records'),len(b),fair=True,transport=True)
    y=b.copy();union=np.zeros(len(b),bool);occupied=np.zeros(len(b),bool);event_id=np.full(len(b),-1);coremask=np.zeros(len(b),int);rows=[]
    selected={candidates[j]['request_index']:j for j in chosen}
    for i,req in enumerate(requests.to_dict('records')):
        row=dict(**req,success=i in selected)
        if i in selected:
            j=selected[i];c=candidates[j];s=c['start'];h=c['width'];patch,x,added=cache[j]
            assert not occupied[c['left']:c['right']].any();occupied[c['left']:c['right']]=True
            y[s-h:s+36+h]=patch;union[s-h:s+36+h]=True;event_id[s:s+36]=i;coremask[s:s+36]=mask[c['bank_index']]
            ce=float(np.max(abs(y[s:s+36][mask[c['bank_index']]]-x[mask[c['bank_index']]]),initial=0));ne=float(np.max(abs((y[s:s+36]-x)@net)))
            edges=np.r_[np.arange(s-h,s+1),np.arange(s+36,s+36+h+1)]
            re=max(float(np.max(abs(y[edges]-y[edges-1])/ramp)-1),float(np.max(abs((y[edges]-y[edges-1])@net)/nr)-1))
            assert ce<1e-6 and ne<1e-6*scale[0] and re<1e-6
            row.update(**c,core_error=ce,net_error=ne,ramp_residual=re,added_risk=added)
        rows.append(row)
    assert np.array_equal(y[~union],b[~union])
    annual=pd.DataFrame(y,columns=['load_kw','wind_kw','pv_kw']);annual.insert(0,'timestamp',times)
    annual['event_id']=event_id;annual['event_core']=coremask;annual['is_extreme_condition']=(event_id>=0).astype(int)
    annual['random_sequence_id']=f'paper_{args.dataset}_{args.seed}';annual['sequence_weight']=1.
    annual.to_csv(out/'annual_source_load.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'events.csv',index=False);pd.DataFrame(candidates).to_csv(out/'candidates.csv',index=False);pd.DataFrame(failures).to_csv(out/'infeasible.csv',index=False)
    (out/'optimization_certificate.json').write_text(json.dumps(proof,indent=2))
    inputs=[data/'fitted_parameters.json',data/'meta_train.csv',data/'event_mask_train.npy',bank/'planning_conditions.csv',bank/'planning.npy',Path(fitted['source'])]
    inputs+=list(paper.glob('annual*.py'))+list(paper.glob('publication*.py'))
    summary=dict(method='SC-RCRB + historical calendar + certified LP + adaptive_fair_transport',dataset=args.dataset,year=args.year,seed=args.seed,requested=len(rows),embedded=len(chosen),complete=len(rows)==len(chosen),generation='fresh inference from frozen paper model' if args.regenerate else 'existing paper candidate bank',common_kw_scale=factor,probability_scope='reference-data conditional stress simulation; not Alashankou annual probability',sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'provenance.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='sha256'})
if __name__=='__main__':main()
