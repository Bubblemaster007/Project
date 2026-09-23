"""Connect certified paper source-load arrays to the existing project demonstrator."""
from pathlib import Path
import subprocess,sys,json
import numpy as np
import pandas as pd

def run_current_pipeline(config_path,config):
    root=Path(__file__).resolve().parents[2]
    paper=config['paper_method'];dataset=paper['dataset'];year=paper['year'];seed=paper['seed']
    cmd=[sys.executable,str(root/'scripts/run_current_paper_method.py'),'--paper-root',paper['root'],'--dataset',dataset,'--year',str(year),'--seed',str(seed)]
    if paper.get('regenerate',True):cmd.append('--regenerate')
    subprocess.run(cmd,check=True)
    out=root/'outputs/current_paper'/dataset/str(year)/f'seed{seed}'
    annual=pd.read_csv(out/'annual_source_load.csv',parse_dates=['timestamp'])
    from src.chapter3.lankao_topology_adapter import build_device_params_from_case33
    from src.chapter3.extreme_generator_adapter import build_hazard_from_scene
    from src.chapter4.balance_analyzer_adapter import run_balance_analysis
    from src.common.io_utils import write_json
    sys.path.insert(0,str(root/'topic2_remaining_code'))
    from fault_probability_model import build_device_state_sequence,FaultModelConfig
    from coupled_condition_builder import build_coupled_condition
    from strategy_trigger import trigger_strategies,StrategyThresholds
    from resilience_reliability_planner import run_simple_planning,PlanningConfig
    devices,_,_,_=build_device_params_from_case33(root/'data/case33')
    for kind,value in [('grid_channel',4000),('transformer',5000),('storage',500),('emergency_gen',500),('renewable',2500)]:
        devices.loc[devices.device_type.eq(kind),'rated_kw']=value
    coupled=annual.copy();coupled['available_re_kw']=coupled.wind_kw+coupled.pv_kw;coupled['reachable_load_kw']=coupled.load_kw
    state_rows=[];hazards=[]
    for eid,scene in annual[annual.event_id.ge(0)].groupby('event_id',sort=False):
        scene=scene.reset_index(drop=True);hazard=build_hazard_from_scene(scene,seed+int(eid));hazard['event_id']=eid;hazards.append(hazard)
        states,capacity=build_device_state_sequence(hazard,devices,FaultModelConfig(random_seed=seed+int(eid)))
        states['event_id']=eid;state_rows.append(states)
        event=build_coupled_condition(scene,capacity)
        idx=coupled.index[coupled.event_id.eq(eid)]
        for col in event.columns:
            if col not in coupled:coupled[col]=np.nan
            coupled.loc[idx,col]=event[col].to_numpy()
    # Certified source trajectories must not be smoothed or embedded again.
    assert np.array_equal(coupled[['load_kw','wind_kw','pv_kw']],annual[['load_kw','wind_kw','pv_kw']])
    coupled.to_csv(out/'annual_random_production_sequence.csv',index=False)
    pd.concat(state_rows).to_csv(out/'device_states.csv',index=False);pd.concat(hazards).to_csv(out/'synthetic_hazards.csv',index=False)
    metrics,hourly,paths=run_balance_analysis(coupled,out/'balance',config['chapter4'])
    triggers=trigger_strategies(metrics,StrategyThresholds(**config.get('strategy_thresholds',{})));triggers.to_csv(out/'strategies.csv',index=False)
    plan=run_simple_planning(metrics,triggers,PlanningConfig(**config.get('planning',{})));write_json(out/'planning.json',plan)
    provenance=json.loads((out/'provenance.json').read_text())
    summary=dict(config=str(config_path),method=provenance['method'],embedding_complete=provenance['complete'],requested=provenance['requested'],embedded=provenance['embedded'],limitations=['Reference singleton data, not Alashankou weather','Synthetic hazards; existing independent-hour device-state demonstration','Aggregated balance, not nodal outage power flow','Heuristic planning, not opportunity-constrained optimization'],chapter3={'annual_random_production_sequence':str(out/'annual_random_production_sequence.csv')},chapter4=paths,chapter6={'planning_result_json':str(out/'planning.json')},metrics=metrics)
    write_json(out/'run_summary.json',summary)
    return summary
