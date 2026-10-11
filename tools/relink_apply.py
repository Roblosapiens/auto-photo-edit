import sqlite3, json, sys, shutil, subprocess, time, os

cat, planfile = sys.argv[1], sys.argv[2]
EXTRA = [("J:/Dropbox/#Photos/#2019/", "J:/Dropbox/#Photos/ALL PHOTOS/#2019/")]

if "--real" in sys.argv:
    tasks = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    if "lightroom.exe" in tasks:
        sys.exit("Lightroom is still running - close it first.")
    backup = cat.replace(".lrcat", f" BACKUP {time.strftime('%Y%m%d-%H%M%S')}.lrcat")
    shutil.copy2(cat, backup)
    print("backup:", backup)

pairs = [(p["old"], p["new"]) for p in json.load(open(planfile))["plan"]] + EXTRA
con = sqlite3.connect(cat)
n = 0
for old, new in pairs:
    assert os.path.isdir(new), new
    name = new.rstrip("/").split("/")[-1]
    n += con.execute("UPDATE AgLibraryRootFolder SET absolutePath=?, name=? WHERE absolutePath=?",
                     (new, name, old)).rowcount
con.commit()
print(f"updated {n} of {len(pairs)} root folders")
print("integrity:", con.execute("PRAGMA integrity_check").fetchone()[0])
