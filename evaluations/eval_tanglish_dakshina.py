"""Score Tamil->Tanglish converters against Dakshina (human romanisation).
Word counts as right if it matches the person's spelling OR any accepted
variant in Dakshina's lexicon."""
import ast, json, re, sys, random
import jiwer, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pipeline.tanglish import convert

p = hf_hub_download("Anvesh-Lankala/Copy_Dakshina_Google_research_dataset", "data/ta-00000-of-00001.parquet", repo_type="dataset")
rows = pq.read_table(p).to_pylist()
random.Random(0).shuffle(rows)
rows = [r for r in rows if r["native_word_count"] == r["romanized_word_count"]][:1500]
norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
for method in sys.argv[1:] or ["plain", "rules", "rules+en", "auto"]:
    ok = tot = 0; cer_r, cer_h = [], []
    for r in rows:
        try:
            lex = json.loads(r["lexicon_mapping"])
            align = r["token_alignment"]
        except Exception:
            continue
        for a in align:
            nt, rt = a["native_token"], a["romanized_token"]
            if not re.search(r"[஀-௿]", nt):
                continue
            nt_clean = re.sub(r"[^஀-௿]", "", nt)
            hyp = norm(convert(nt_clean, method))
            accepted = {norm(rt)} | {norm(v) for v in lex.get(nt_clean, [])}
            ok += hyp in accepted; tot += 1
            cer_r.append(norm(rt)); cer_h.append(hyp)
    print(f"{method:9s} word matches a human spelling: {100*ok/tot:5.1f}%   char error vs the person's spelling: {100*jiwer.cer(' '.join(cer_r), ' '.join(cer_h)):5.1f}%   ({tot} words)")
