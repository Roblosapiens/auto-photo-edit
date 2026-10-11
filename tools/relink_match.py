import sqlite3, os, sys, json, collections

cat, out = sys.argv[1], sys.argv[2]
SEARCH = "J:/Dropbox/#Photos/"   # search all of #Photos, ALL PHOTOS included

con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True)
roots = con.execute("SELECT id_local, absolutePath FROM AgLibraryRootFolder").fetchall()
# files per root: (pathFromRoot, filename)
rows = con.execute("""SELECT fo.rootFolder, fo.pathFromRoot, fi.idx_filename
                      FROM AgLibraryFile fi JOIN AgLibraryFolder fo ON fi.folder = fo.id_local""").fetchall()
files = collections.defaultdict(list)
for rid, p, fn in rows:
    files[rid].append((p, fn))

# index every directory under SEARCH by lowercase leaf name
index = collections.defaultdict(list)
for dp, dns, _ in os.walk(SEARCH):
    for d in dns:
        index[d.lower()].append(os.path.join(dp, d).replace("\\", "/") + "/")

def score(cand, rid):
    fl = files[rid]
    sample = fl[:: max(1, len(fl) // 40)]
    if not sample:
        return 0.0
    return sum(os.path.isfile(cand + p + fn) for p, fn in sample) / len(sample)

plan, unresolved = [], []
for rid, ap in roots:
    if os.path.isdir(ap):
        continue
    leaf = ap.rstrip("/").split("/")[-1].lower()
    cands = [(score(c, rid), c) for c in index.get(leaf, [])]
    cands.sort(reverse=True)
    if cands and cands[0][0] >= 0.8 and (len(cands) == 1 or cands[1][0] < cands[0][0]):
        plan.append({"id": rid, "old": ap, "new": cands[0][1], "score": round(cands[0][0], 2)})
    else:
        unresolved.append({"old": ap, "nfiles": len(files[rid]), "cands": [(round(s, 2), c) for s, c in cands[:3]]})

json.dump({"plan": plan, "unresolved": unresolved}, open(out, "w"), indent=1)
print(f"matched {len(plan)} roots ({sum(len(files[p['id']]) for p in plan)} photos); unresolved {len(unresolved)}")
for p in plan:
    if p["score"] < 1:
        print(f"  partial {p['score']}: {p['old']} -> {p['new']}")
print("UNRESOLVED:")
for u in unresolved:
    print(f"  {u['nfiles']:5d} {u['old']}  cands={u['cands']}")
