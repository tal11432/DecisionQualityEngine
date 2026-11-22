"""Side-by-side headline metrics of the committed (legacy-population) reports and the set-piece-excluded reruns.
Usage: python research/set_piece_check/compare_runs.py [BASE_DIR=reports] [EXCL_DIR=reports/set_piece_exclusion]
Parses cross_game_results*.txt and pooled_validation_results.txt; '*' = bootstrap CI excludes 0 (as in the source files)."""
import re, sys
DIRS = {"base": sys.argv[1] if len(sys.argv) > 1 else "reports", "excl": sys.argv[2] if len(sys.argv) > 2 else "reports/set_piece_exclusion"}
def load(d,f): return open(f"{DIRS[d]}/{f}").read().split("\n")

def pooled(lines):
    out={}; sc=None
    for i,l in enumerate(lines):
        m=re.match(r"SCENARIO ([AB]):",l)
        if m: sc=m.group(1)
        m=re.match(r"--- ([AB]): train .*-> test (G\d) \(n=(\d+), (\d+) intercepted / (\d+) completed; rate ([\d.]+)\) ---",l)
        if m:
            d={"n":int(m.group(3)),"int":int(m.group(4)),"rate":float(m.group(6))}
            for k in range(i+1,i+8):
                mm=re.match(r"(M2|M3|M3x)\s.*?\s([\d.]+) \[([\d.]+), ([\d.]+)\]\s+([\d.]+)\s+([+-][\d.]+) \[[^\]]*\](\*?)\s+([+-][\d.]+) \[[^\]]*\](\*?)\s+([\d.-]+) \[",lines[k])
                if mm:
                    key=mm.group(1)
                    d[key]=dict(auc=float(mm.group(2)),dB=float(mm.group(6)),dBs=mm.group(7),dL=float(mm.group(8)),dLs=mm.group(9),slope=float(mm.group(10)))
            out[(m.group(1),m.group(2))]=d
    return out

def cross(lines):
    out={}; sc=None
    for i,l in enumerate(lines):
        m=re.match(r"SCENARIO ([AB]):",l)
        if m: sc=m.group(1)
        m=re.match(r"(G\d) -> (G\d): n=(\d+) \((\d+) intercepted; test completion rate ([\d.]+)",l)
        if m and sc:
            d={"n":int(m.group(3)),"int":int(m.group(4))}
            for k in range(i+1,i+6):
                mm=re.match(r"\s+(PRIMARY logistic|boosted trees \(exploratory\))\s+ROC-AUC ([\d.]+) \[[^\]]*\]\s+PR-AUC ([\d.]+)\s+dBrier ([+-][\d.]+) \[[^\]]*\](\*?)\s+dLogLoss ([+-][\d.]+) \[[^\]]*\](\*?)\s+-> (.*)",lines[k])
                if mm:
                    d[mm.group(1).split()[0]]=dict(auc=float(mm.group(2)),dB=float(mm.group(4)),dBs=mm.group(5),dL=float(mm.group(6)),dLs=mm.group(7),verdict=mm.group(8))
            out[(sc,m.group(1)+">"+m.group(2))]=d
    return out
def skill(lines):
    r={}; sc=None
    for l in lines:
        m=re.match(r"SCENARIO ([AB]):",l)
        if m: sc=m.group(1)
        m=re.match(r"(feasibility baseline|primary logistic|boosted trees)\s+prob.skill yes (\d), partial (\d), no (\d)",l)
        if m: r[(sc,m.group(1))]=tuple(map(int,m.groups()[1:]))
    return r

def table_cross(f):
    b,e=cross(load("base",f)),cross(load("excl",f))
    for sc in "AB":
        print(f"\n[{f}] scenario {sc}: PRIMARY logistic (train>test): n int | AUC | dBrier | dLogLoss  (base -> excl; * = CI excludes 0)")
        for k in [k for k in b if k[0]==sc]:
            x,y=b[k],e[k]; p,q=x["PRIMARY"],y["PRIMARY"]
            print(f"  {k[1]}  n {x['n']}->{y['n']}  int {x['int']}->{y['int']} | AUC {p['auc']:.3f}->{q['auc']:.3f} ({q['auc']-p['auc']:+.3f}) | dBrier {p['dB']:+.4f}{p['dBs']}->{q['dB']:+.4f}{q['dBs']} | dLL {p['dL']:+.4f}{p['dLs']}->{q['dL']:+.4f}{q['dLs']} | {p['verdict']} -> {q['verdict']}")
        import statistics as st
        ks=[k for k in b if k[0]==sc]
        print("  mean primary AUC over 6 runs: %.3f -> %.3f ; mean boosted AUC %.3f -> %.3f"%(st.mean(b[k]['PRIMARY']['auc'] for k in ks),st.mean(e[k]['PRIMARY']['auc'] for k in ks),st.mean(b[k]['boosted']['auc'] for k in ks),st.mean(e[k]['boosted']['auc'] for k in ks)))
    sb,se=skill(load("base",f)),skill(load("excl",f))
    for k in sb: print("  skill",k,sb[k],"->",se[k])

for f in ("cross_game_results.txt","cross_game_results_fix_identity.txt"): table_cross(f)

b,e=pooled(load("base","pooled_validation_results.txt")),pooled(load("excl","pooled_validation_results.txt"))
print("\n[pooled_validation_results.txt]  M3 (single score) and M2: AUC | dBrier | dLogLoss | cal slope")
for k in b:
    x,y=b[k],e[k]
    print(f" {k[0]} test {k[1]}: n {x['n']}->{y['n']} int {x['int']}->{y['int']} rate {x['rate']:.3f}->{y['rate']:.3f}")
    for m in ("M2","M3"):
        p,q=x[m],y[m]
        print(f"     {m}: AUC {p['auc']:.3f}->{q['auc']:.3f} | dB {p['dB']:+.4f}{p['dBs']}->{q['dB']:+.4f}{q['dBs']} | dLL {p['dL']:+.4f}{p['dLs']}->{q['dL']:+.4f}{q['dLs']} | slope {p['slope']:.2f}->{q['slope']:.2f}")
