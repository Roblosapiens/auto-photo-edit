"""Run the Lightroom plugin's Lua against a fake Lightroom SDK (tests/fake_lightroom.lua).

Usage: python tests/test_plugin.py   (needs `pip install lupa`)
"""
from pathlib import Path

from lupa import LuaRuntime

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "lightroom" / "AutoPhotoEdit.lrplugin"


def setup():
    lua = LuaRuntime(unpack_returned_tuples=True)
    for f in PLUGIN.glob("*.lua"):
        assert lua.execute("return function(s) return load(s) ~= nil end")(f.read_text(encoding="utf-8")), f"syntax: {f.name}"
    lua.execute((ROOT / "tests" / "fake_lightroom.lua").read_text(encoding="utf-8"))
    core = lua.execute((PLUGIN / "AutoEditCore.lua").read_text(encoding="utf-8"))
    lua.globals().core = core
    return lua, lua.eval("function(mode) core.start(mode) end")


def test_normal_then_redo():
    lua, run = setup()
    photo = lambda i: lua.eval("photos")[i]
    run("crop")
    assert photo(1).dev.Exposure2012 == 1.5 and photo(1).props.autoEditMode is None, "key photo untouched"
    assert [photo(i).dev.CropLeft for i in (2, 3, 4, 5)] == [0.1, 0.1, 0.1, None]
    assert all(photo(i).props.autoEditDate for i in (2, 3, 4, 5)), "records written"

    # hand edits: tone on 3, adjusted plugin crop on 4, own crop on 5 (plugin left it uncropped)
    lua.execute("photos[3].dev.Exposure2012 = 0.5; photos[4].dev.CropLeft = 0.25; "
                "photos[5].dev.CropLeft = 0.3; photos[5].dev.CropRight = 0.7")

    lua.execute('EXPO_SHIFT = 0.2; answers = {"ok", "ok"}')  # keep my changes
    run("redo")
    assert photo(3).dev.Exposure2012 == 0.5, "tone hand edit kept"
    assert photo(4).dev.CropLeft == 0.25 and photo(5).dev.CropLeft == 0.3, "hand crops kept"
    assert abs(photo(2).dev.Exposure2012 - 1.4) < 1e-9 and abs(photo(4).dev.Exposure2012 - 1.6) < 1e-9, "tone redone"

    lua.execute('EXPO_SHIFT = 0.4; answers = {"other", "ok"}')  # redo them too
    run("redo")
    assert abs(photo(3).dev.Exposure2012 - 1.7) < 1e-9 and photo(3).dev.CropLeft == 0.1, "photo 3 fully redone"
    assert photo(4).dev.CropLeft == 0.25 and photo(5).dev.CropLeft == 0.3, "crop-only hand edits still kept"

    lua.execute("log = {}")
    run("redo")
    log = lua.eval("log")
    assert any("nothing to change" in log[i] for i in range(1, len(log) + 1)), "no-op redo says so"


if __name__ == "__main__":
    test_normal_then_redo()
    print("plugin tests passed")
