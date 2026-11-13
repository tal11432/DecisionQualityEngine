import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.config import load_config
from src.cli import _load
from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.pass_completion_features import _Track
from src.pass_dataset import DecisionContract
from src.processing.sync import synchronize_events
cfg = load_config(); tol = cfg["sync"]["tolerance_s"]; C = DecisionContract.from_config(cfg); P = FeasibilityParams.from_config(cfg); W=C.velocity_window_frames
GX=52.5
rows=[]; crow=[]
for g in (1,2,3):
    td, ev = _load(cfg, g)
    sh = ev[ev.type=="SHOT"].copy()
    T=_Track.of(td)
    m = synchronize_events(sh[["team","period","start_time_s","from_player"]], td.frames, tol, "start_time_s", None)
    mm = m.matched.set_index("event_idx")["frame_row"]
    for idx,e in sh.iterrows():
        sub=str(e.subtype); head="HEAD" in sub
        # preceding event
        prev = ev[(ev.index<idx)].tail(1)
        prev_t = (prev.type.iloc[0], str(prev.subtype.iloc[0]), prev.team.iloc[0]==e.team, e.start_time_s-prev.end_time_s.iloc[0]) if len(prev) else (None,)*4
        r=dict(game=g,idx=idx,sub=sub,head=head,goal="GOAL" in sub,prev_type=prev_t[0],prev_sub=prev_t[1],prev_same=prev_t[2],gap=prev_t[3],synced=idx in mm.index)
        if idx in mm.index:
            r0=int(mm[idx]); team=e.team
            try: j=next(k for k in range(len(T.pid)) if T.team[k]==team and T.pid[k]==e.from_player)
            except StopIteration: r["synced"]=False; rows.append(r); continue
            sg=T.sign(team,r0)
            origin=np.array([T.X[r0,j],T.Y[r0,j]])
            r["ball_d"]=float(np.hypot(T.X[r0,j]-T.bx[r0],T.Y[r0,j]-T.by[r0])) if True else np.nan
            ok=np.isfinite(T.X[r0])&np.isfinite(T.Y[r0])
            mates=np.flatnonzero((T.team==team)&ok&(np.arange(len(T.pid))!=j)); opp=np.flatnonzero((T.team!=team)&ok)
            opp_xy=np.column_stack([T.X[r0,opp],T.Y[r0,opp]])
            ox=np.sort(opp_xy[:,0]*sg); sl=ox[-2] if len(ox)>1 else ox[-1]; line=max(sl,origin[0]*sg)
            opp_v=np.column_stack([T.X[r0,opp]-T.X[r0-W,opp],T.Y[r0,opp]-T.Y[r0-W,opp]])/float(T.time[r0]-T.time[r0-W])
            sx,sy=origin[0]*sg,origin[1]*sg
            r.update(sx=sx,sy=sy,shot_dist=float(np.hypot(GX-sx,sy)),n_mates=len(mates),n_opp=len(opp),
                     shooter_press=float(np.hypot(*(opp_xy-origin).T).min()))
            # goal-side opponents between shooter and goal (rough: opponents with x>shooter x within 7.32m corridor)
            r["period"]=int(T.period[r0])
            nc0=nc_att=nc_near=nc_safe=nc_better=0
            for mt in mates:
                xy=np.array([T.X[r0,mt],T.Y[r0,mt]]); d=float(np.hypot(*(xy-origin))); xo,yo=xy[0]*sg,xy[1]*sg
                if not (3<=d<=40): continue
                if xo>line and xo>0: continue
                nc0+=1
                sd=float(np.hypot(GX-xo,yo))
                if sd<=35: nc_att+=1
                mg=interception_margin(origin,xy,opp_xy,P,opp_v)["margin_s"]
                crow.append(dict(game=g,idx=idx,cand=T.pid[mt],d=d,fwd=xo-sx,shot_d=sd,margin=float(mg),rec_press=float(np.hypot(*(opp_xy-xy).T).min()),closer=sd<r["shot_dist"]))
            r.update(nC0=nc0,nC_att=nc_att)
        rows.append(r)
S=pd.DataFrame(rows); Cd=pd.DataFrame(crow)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(S.groupby("game").agg(n=("idx","size"),heads=("head","sum"),goals=("goal","sum"),synced=("synced","sum")))
print(S[["game","sub","prev_type","prev_sub","prev_same","gap","shot_dist","shooter_press","n_mates","nC0","nC_att"]].to_string())
S.to_pickle("research/shots/shots.pkl"); Cd.to_pickle("research/shots/shotc.pkl")
