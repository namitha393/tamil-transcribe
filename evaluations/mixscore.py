import json, sys
from normalize import normalize
def found(kind, model):
    try: rows=[json.loads(l) for l in open(f"results/{kind}/{model}.jsonl")]
    except FileNotFoundError: return None
    if len(rows) < sum(1 for _ in open(f"data/{kind}/manifest.jsonl")): return None
    hit=tot=0
    for r in rows:
        hw=set(normalize(r["hyp"]).split()); aw=normalize(r["ref"]).split(); hit+=sum(w in hw for w in aw); tot+=len(aw)
    return 100*hit/tot
for m in sys.argv[1:]:
    print(m, {k: (round(v,1) if v is not None else "pending") for k in ["mix_wide","alone_phone","ctl_other_speaker"] for v in [found(k,m)]})
