-- Fake Lightroom SDK for testing AutoEditCore.lua outside Lightroom
TMP = os.getenv('TEMP') .. '/autoedit_test'
os.execute('mkdir "' .. TMP .. '" 2>nul')
answers = {}
log = {}
photos = {}
local function mk(id, label, dev)
  local p = {localIdentifier = id, label = label, dev = dev, props = {}}
  function p:getDevelopSettings() local c = {} for k, v in pairs(self.dev) do c[k] = v end return c end
  function p:applyDevelopSettings(s) for k, v in pairs(s) do self.dev[k] = v end end
  function p:getPropertyForPlugin(_, k) return self.props[k] end
  function p:setPropertyForPlugin(_, k, v) self.props[k] = v end
  photos[#photos + 1] = p
end
mk(1, 'green', {Exposure2012 = 1.5, Temperature = 5000, WhiteBalance = 'Custom', CropLeft = 0.2, CropRight = 0.8})
mk(2, 'none', {}) mk(3, 'none', {}) mk(4, 'none', {}) mk(5, 'none', {})

-- fake predict.py: tone for 2..5; crops for 2..4 only when the request says "not cropped"
function fake_python(reqPath, respPath)
  local req = io.open(reqPath):read('*a')
  local out = io.open(respPath, 'w')
  for id = 2, 5 do
    out:write(id .. '\tExposure2012\t' .. (1.0 + id / 10 + (EXPO_SHIFT or 0)) .. '\n')
    out:write(id .. '\tWhiteBalance\tCustom\n')
    local obj
    for o in req:gmatch('{[^{}]*}') do if o:find('"id":' .. id .. '[,}]') then obj = o end end
    if id <= 4 and obj and obj:find('"cropped":false') then
      for _, kv in ipairs({{'CropLeft', 0.1}, {'CropRight', 0.9}, {'CropTop', 0.05}, {'CropBottom', 0.95}, {'CropAngle', 1.2}}) do
        out:write(id .. '\t' .. kv[1] .. '\t' .. kv[2] .. '\n')
      end
      out:write(id .. '\tCropConstrainAspectRatio\tfalse\n')
    end
  end
  out:close()
  if SKIP_ID then
    local sk = io.open(respPath .. '.skipped', 'w'); sk:write(SKIP_ID .. '\tOSError: image file is truncated\n'); sk:close()
  end
end

local catalog = {}
function catalog:getTargetPhotos() return photos end
function catalog:batchGetRawMetadata(ps)
  local t = {} for _, p in ipairs(ps) do t[p] = {path = 'X:/p' .. p.localIdentifier .. '.dng', fileFormat = 'DNG', colorNameForLabel = p.label} end return t
end
function catalog:batchGetFormattedMetadata(ps) local t = {} for _, p in ipairs(ps) do t[p] = {flash = 'Did not fire'} end return t end
function catalog:withWriteAccessDo(name, fn) log[#log + 1] = 'undo step: ' .. name; fn() end

local mods = {
  LrApplication = {activeCatalog = function() return catalog end},
  LrDialogs = {
    message = function(t, m) log[#log + 1] = 'MESSAGE ' .. t .. ' | ' .. (m or '') end,
    confirm = function(t, m) local a = table.remove(answers, 1) or 'ok'; log[#log + 1] = 'CONFIRM ' .. t .. ' | ' .. m:gsub('\n', ' / ') .. '  -> ' .. a; return a end,
    attachErrorDialogToFunctionContext = function() end,
    showBezel = function(t) log[#log + 1] = 'BEZEL ' .. t end,
  },
  LrFunctionContext = {callWithContext = function(_, fn) fn({}) end},
  LrPathUtils = {
    parent = function(p) return p end,
    leafName = function(p) return p:match('[^/\\]+$') end,
    child = function(a, b) if a == 'TMPDIR' then return TMP .. '/' .. b end return a .. '/' .. b end,
    getStandardFilePath = function() return 'TMPDIR' end,
  },
  LrFileUtils = {
    exists = function(p) if p:find('python') then return true end local f = io.open(p) if f then f:close() return true end return false end,
    delete = function(p) os.remove(p) end,
  },
  LrProgressScope = function() return {setCaption = function() end, setPortionComplete = function() end, done = function() end} end,
  LrTasks = {
    startAsyncTask = function(fn) fn() end,
    sleep = function() end,
    execute = function(cmd)
      local req, resp = cmd:match('python%.exe" "[^"]-" "([^"]-)" "([^"]-)"')
      fake_python(req, resp)
      return 0
    end,
  },
}
import = function(name) return mods[name] end
_PLUGIN = {path = 'PLUGIN'}
