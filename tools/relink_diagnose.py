import sqlite3, os, sys, collections

cat = sys.argv[1]
ALL = "J:/Dropbox/#Photos/ALL PHOTOS/"
OLD_PREFIX = "J:/Dropbox/#Photos/"

con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True)
roots = con.execute("SELECT id_local, absolutePath, name FROM AgLibraryRootFolder").fetchall()
folders = con.execute("SELECT id_local, rootFolder, pathFromRoot FROM AgLibraryFolder").fetchall()
counts = dict(con.execute("SELECT folder, COUNT(*) FROM AgLibraryFile GROUP BY folder").fetchall())

print(f"roots={len(roots)} folders={len(folders)}")
by_root = collections.defaultdict(list)
for fid, rid, p in folders:
    by_root[rid].append((fid, p))

missing_roots = []
for rid, ap, name in roots:
    exists = os.path.isdir(ap)
    nfiles = sum(counts.get(fid, 0) for fid, _ in by_root[rid])
    if not exists:
        cand = None
        if ap.startswith(OLD_PREFIX) and not ap.startswith(ALL):
            c = ALL + ap[len(OLD_PREFIX):]
            if os.path.isdir(c):
                cand = c
        missing_roots.append((ap, nfiles, cand))

print(f"\nMISSING ROOT FOLDERS: {len(missing_roots)}")
fixable = [m for m in missing_roots if m[2]]
print(f"  auto-fixable via ALL PHOTOS: {len(fixable)}  ({sum(m[1] for m in fixable)} photos)")
for ap, n, cand in missing_roots:
    print(f"  {'OK ' if cand else '?? '} {n:6d}  {ap}" + (f"  ->  {cand}" if cand else ""))

# subfolders missing under roots that DO exist
present = {rid: ap for rid, ap, _ in roots if os.path.isdir(ap)}
sub_missing = [(present[rid] + p, counts.get(fid, 0)) for fid, rid, p in folders
               if rid in present and p and not os.path.isdir(present[rid] + p)]
print(f"\nMISSING SUBFOLDERS under existing roots: {len(sub_missing)}")
for path, n in sub_missing[:40]:
    print(f"  {n:6d}  {path}")
