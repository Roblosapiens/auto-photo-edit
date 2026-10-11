--[[
Auto Edit from Key Photos

Select every photo of one event, give the 1-3 photos you edited by hand a green label, then run
Library > Plug-in Extras > Auto Edit from Key Photos. The other photos get the key photos' settings,
with exposure and white balance adjusted per photo by the model (src/predict.py).
Everything happens in one undo step (Ctrl+Z).
]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFunctionContext = import 'LrFunctionContext'
local LrPathUtils = import 'LrPathUtils'
local LrFileUtils = import 'LrFileUtils'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local KEY_LABEL = 'green'
local RAW_FORMATS = { RAW = true, DNG = true }

-- The plugin lives in <repo>/lightroom/AutoPhotoEdit.lrplugin
local REPO = LrPathUtils.parent(LrPathUtils.parent(_PLUGIN.path))
local PYTHON = LrPathUtils.child(REPO, '.venv\\Scripts\\python.exe')
local SCRIPT = LrPathUtils.child(REPO, 'src\\predict.py')

-- Minimal JSON encoding (Lightroom has no JSON library) ---------------------------------------

local function jsonString(s)
	s = tostring(s):gsub('\\', '\\\\'):gsub('"', '\\"')
	return '"' .. s .. '"'
end

local function jsonValue(v)
	local t = type(v)
	if t == 'number' then
		if v ~= v or v == math.huge or v == -math.huge then return 'null' end
		return string.format('%.10g', v)
	elseif t == 'boolean' then
		return tostring(v)
	elseif t == 'string' then
		return jsonString(v)
	end
	return 'null'
end

-- Encode the scalar fields of a table; nested tables (tone curves etc.) are skipped.
local function jsonObject(tbl, extra)
	local parts = {}
	for k, v in pairs(tbl) do
		if type(v) ~= 'table' then
			parts[#parts + 1] = jsonString(k) .. ':' .. jsonValue(v)
		end
	end
	for k, v in pairs(extra or {}) do
		parts[#parts + 1] = jsonString(k) .. ':' .. v
	end
	return '{' .. table.concat(parts, ',') .. '}'
end

-- Helpers --------------------------------------------------------------------------------------

local function readFile(path)
	local f = io.open(path, 'r')
	if not f then return '' end
	local s = f:read('*a')
	f:close()
	return s
end

local function tail(s, n)
	return #s > n and ('...' .. s:sub(-n)) or s
end

local function parseResponse(path)
	local settings, count = {}, 0
	for line in io.lines(path) do
		local id, key, val = line:match('^(%d+)\t([^\t]+)\t(.-)%s*$')
		if id then
			id = tonumber(id)
			if not settings[id] then
				settings[id] = {}
				count = count + 1
			end
			settings[id][key] = tonumber(val) or val
		end
	end
	return settings, count
end

-- Main -----------------------------------------------------------------------------------------

local function run(context)
	local catalog = LrApplication.activeCatalog()
	local selected = catalog:getTargetPhotos()

	local meta = catalog:batchGetRawMetadata(selected, {
		'path', 'fileFormat', 'colorNameForLabel', 'isoSpeedRating', 'shutterSpeed', 'aperture',
		'focalLength', 'dateTimeOriginal',
	})
	local flashText = catalog:batchGetFormattedMetadata(selected, { 'flash' })

	local photos, byId, records, nKeys, nSkipped = {}, {}, {}, 0, 0
	for _, photo in ipairs(selected) do
		local m = meta[photo]
		if RAW_FORMATS[m.fileFormat] then
			local isKey = m.colorNameForLabel == KEY_LABEL
			local flash = (flashText[photo].flash or ''):lower()
			local rec = {
				id = photo.localIdentifier, path = m.path, key = isKey,
				iso = m.isoSpeedRating, shutter = m.shutterSpeed, aperture = m.aperture,
				focal = m.focalLength, time = m.dateTimeOriginal,
				flash = flash:find('fire') ~= nil and flash:find('not') == nil,
			}
			local extra = nil
			if isKey then
				nKeys = nKeys + 1
				extra = { settings = jsonObject(photo:getDevelopSettings()) }
			end
			records[#records + 1] = jsonObject(rec, extra)
			photos[#photos + 1] = photo
			byId[photo.localIdentifier] = photo
		else
			nSkipped = nSkipped + 1
		end
	end

	if nKeys == 0 then
		LrDialogs.message('Auto Edit: no key photos',
			'Give the 1-3 photos you edited by hand a green label (press 8), select all photos of the event, and run this again.',
			'info')
		return
	end
	if #photos <= nKeys then
		LrDialogs.message('Auto Edit', 'Select the other photos of the event as well as the green-labelled key photos.', 'info')
		return
	end
	if not LrFileUtils.exists(PYTHON) then
		LrDialogs.message('Auto Edit: Python not found', 'Expected the project environment at\n' .. PYTHON, 'critical')
		return
	end

	local tmp = LrPathUtils.getStandardFilePath('temp')
	local reqPath = LrPathUtils.child(tmp, 'autoedit_request.json')
	local respPath = LrPathUtils.child(tmp, 'autoedit_response.tsv')
	local logPath = LrPathUtils.child(tmp, 'autoedit_log.txt')
	LrFileUtils.delete(respPath)

	local f = io.open(reqPath, 'w')
	f:write('{"photos":[' .. table.concat(records, ',\n') .. ']}')
	f:close()

	local progress = LrProgressScope { title = 'Auto Edit: analysing ' .. #photos .. ' photos', functionContext = context }
	progress:setCaption('Running the model (about 10-30 seconds)...')
	-- On Windows LrTasks.execute needs the whole command wrapped in an extra pair of quotes
	local cmd = '"' .. '"' .. PYTHON .. '" "' .. SCRIPT .. '" "' .. reqPath .. '" "' .. respPath .. '" > "' .. logPath .. '" 2>&1' .. '"'
	local status = LrTasks.execute(cmd)
	progress:done()

	if status ~= 0 or not LrFileUtils.exists(respPath) then
		LrDialogs.message('Auto Edit failed', tail(readFile(logPath), 1500), 'critical')
		return
	end

	local settings, count = parseResponse(respPath)
	local info = string.format('%d key photo(s) -> %d photo(s) will be updated.', nKeys, count)
	if nSkipped > 0 then
		info = info .. string.format('\n%d non-raw file(s) in the selection are left alone.', nSkipped)
	end
	info = info .. '\nKey photos and crops are not changed. Undo with Ctrl+Z.'
	if LrDialogs.confirm('Apply Auto Edit?', info, 'Apply', 'Cancel') ~= 'ok' then
		return
	end

	catalog:withWriteAccessDo('Auto Edit from Key Photos', function()
		for id, s in pairs(settings) do
			local photo = byId[id]
			if photo then
				photo:applyDevelopSettings(s)
			end
		end
	end, { timeout = 120 })
end

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('AutoEdit', function(context)
		LrDialogs.attachErrorDialogToFunctionContext(context)
		run(context)
	end)
end)
