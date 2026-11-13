import pandas as pd, numpy as np
SP="research/shots/"
S=pd.read_pickle(SP+"shots.pkl"); C=pd.read_pickle(SP+"shotc.pkl")
S["setp"]=S.prev_sub.astype(str).str.contains("PENALTY|FREE KICK")
S["cross"]=S.prev_sub.astype(str).str.contains("CROSS|DEEP BALL")
print("ball_d", S.ball_d.describe().round(2).to_dict(), (S.ball_d>3).sum())
S["step1"]=~S["head"]; S["step2"]=S.step1&~S.setp
S["afterpass"]=S.prev_type.eq("PASS")&S.prev_same&(S.gap<=1.0)
S["cand1"]=S.nC0>=1; S["cand2"]=S.nC0>=2
# per-shot closer & lane-free candidates
cc=C.copy(); cc["closer_free"]=cc.closer&(cc.margin<0); cc["closer_att"]=cc.closer
g=cc.groupby("idx_key" if "idx_key" in cc else ["game","idx"]).agg(n=("cand","size"),n_closer=("closer","sum"),n_cf=("closer_free","sum")).reset_index()
S=S.merge(g,on=["game","idx"],how="left").fillna({"n":0,"n_closer":0,"n_cf":0})
E=S[S.step2]
print("\nBY GAME: all / nonhead / open-play-foot / >=1 cand / >=2 cand / >=1 closer / >=1 closer&margin<0")
for k,d in [*E.groupby("game"),("all",E)]:
    a=S if k=="all" else S[S.game==k]
    print(k,len(a),a.step1.sum(),len(d),(d.nC0>=1).sum(),(d.nC0>=2).sum(),(d.n_closer>=1).sum(),(d.n_cf>=1).sum(), "after-completed-pass<=1s:",d.afterpass.sum(),"goals",d.goal.sum())
E=E.copy(); E["zone"]=pd.cut(E.shot_dist,[0,12,20,50],labels=["<12m","12-20m","20m+"])
print(E.groupby("zone",observed=True).agg(n=("idx","size"),c1=("cand1","sum"),c2=("cand2","sum"),closer=("n_closer",lambda x:(x>=1).sum()),cf=("n_cf",lambda x:(x>=1).sum()),mean_n=("nC0","mean"),mean_closer=("n_closer","mean")).round(2))
print(E.groupby(["game","zone"],observed=True).agg(n=("idx","size"),closer=("n_closer",lambda x:(x>=1).sum()),cf=("n_cf",lambda x:(x>=1).sum())))
print("mean set",E.nC0.mean().round(2),"median closer",E.n_closer.median(),"goals",E.goal.sum(),"shooter_press med",E.shooter_press.median())
print("prev types",E.prev_type.value_counts().to_dict(), "afterpass",E.afterpass.sum())
print("per-shot closer&lane-free dist", E.n_cf.describe().round(2).to_dict())
# example candidates
print(C[C.margin<0].groupby(["game","idx"]).size().head())
S.to_pickle(SP+"shots_full.pkl")
