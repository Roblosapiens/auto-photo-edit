--[[
Shared logic for the Auto Edit menu items.

Select every photo of one event, give the 1-3 photos you edited by hand a green label, then run
Library > Plug-in Extras > Auto Edit .... The other photos get the key photos' settings, with exposure
and white balance adjusted per photo by the model (src/predict.py). With cropping on, photos that are
not cropped yet also get a crop framed around the dancers. Everything is one undo step (Ctrl+Z).

Modes:
  crop  - tone + crop; existing crops are kept
  tone  - tone only
  redo  - like crop, but photos still as the plugin last left them are fully redone (including their
          crop); a crop changed by hand is kept; photos whose tone was changed by hand since are kept
          unless the user chooses to redo them too

Each applied photo gets a record in the catalog (plugin metadata, see MetadataDefinition.lua): the date,
the mode and the exact values applied, which is how redo tells untouched auto edits from hand edits.
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
local MODE_TITLES = { crop = 'Auto Edit + Crop', tone = 'Auto Edit (tone only)', redo = 'Auto Edit + Crop (redo)' }
local CROP_KEYS = { 'CropLeft', 'CropTop', 'CropRight', 'CropBottom', 'CropAngle' }

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

local function isCropped(s)
	return (s.CropLeft or 0) > 0.001 or (s.CropTop or 0) > 0.001 or (s.CropRight or 1) < 0.999
		or (s.CropBottom or 1) < 0.999 or math.abs(s.CropAngle or 0) > 0.01
end

local function parseValue(val)
	if val == 'true' then return true end
	if val == 'false' then return false end
	return tonumber(val) or val
end

-- Lightroom omits settings that are at their default; fill those in before comparing.
local DEFAULTS = { CropLeft = 0, CropTop = 0, CropRight = 1, CropBottom = 1, CropAngle = 0 }

-- True when applying `s` would change anything in the photo's current develop settings.
local function differs(current, s)
	for k, v in pairs(s) do
		local c = current[k]
		if c == nil then c = DEFAULTS[k] end
		if type(v) == 'number' then
			if type(c) ~= 'number' or math.abs(c - v) > 0.001 then return true end
		elseif c ~= v then
			return true
		end
	end
	return false
end

-- Photos predict.py could not read ("<id> TAB <reason>" lines) -> list of ids
local function readSkipped(path)
	local ids = {}
	local f = io.open(path, 'r')
	if not f then return ids end
	for line in f:lines() do
		local id = line:match('^(%d+)\t')
		if id then ids[#ids + 1] = tonumber(id) end
	end
	f:close()
	return ids
end

-- "a.dng, b.dng, c.dng and 4 more"
local function nameList(names, max)
	local shown = {}
	for i = 1, math.min(#names, max) do shown[i] = names[i] end
	local s = table.concat(shown, ', ')
	if #names > max then s = s .. string.format(' and %d more', #names - max) end
	return s
end

local function parseResponse(path)
	local settings, count, crops = {}, 0, 0
	for line in io.lines(path) do
		local id, key, val = line:match('^(%d+)\t([^\t]+)\t(.-)%s*$')
		if id then
			id = tonumber(id)
			if not settings[id] then
				settings[id] = {}
				count = count + 1
			end
			settings[id][key] = parseValue(val)
			if key == 'CropLeft' then crops = crops + 1 end
		end
	end
	return settings, count, crops
end

-- Auto-edit records (catalog plugin metadata) -------------------------------------------------

-- "key=value;key=value", keys sorted, for the hidden autoEditRecord field.
local function serializeRecord(s)
	local keys = {}
	for k in pairs(s) do keys[#keys + 1] = k end
	table.sort(keys)
	local parts = {}
	for _, k in ipairs(keys) do
		local v = s[k]
		parts[#parts + 1] = k .. '=' .. (type(v) == 'number' and string.format('%.6g', v) or tostring(v))
	end
	return table.concat(parts, ';')
end

local function parseRecord(str)
	local s = {}
	for k, v in (str or ''):gmatch('([^;=]+)=([^;]*)') do
		s[k] = parseValue(v)
	end
	return s
end

local IS_CROP_KEY = { CropLeft = true, CropTop = true, CropRight = true, CropBottom = true, CropAngle = true,
	CropConstrainAspectRatio = true }

-- Compare a photo with its record. Returns nil when it was never auto-edited, else two flags:
--   toneChanged  a tone setting differs from what the plugin applied (hand edit, sync, history, reset)
--   cropChanged  the crop differs from the plugin's, or the photo was cropped although the plugin didn't
local function recordState(recordStr, current)
	if not recordStr or recordStr == '' then return nil end
	local tone, crop = {}, {}
	for k, v in pairs(parseRecord(recordStr)) do
		if IS_CROP_KEY[k] then crop[k] = v else tone[k] = v end
	end
	local cropChanged
	if crop.CropLeft ~= nil then
		cropChanged = differs(current, crop)
	else
		cropChanged = isCropped(current)
	end
	return { toneChanged = differs(current, tone), cropChanged = cropChanged }
end

-- Main -----------------------------------------------------------------------------------------

local function run(context, mode)
	local withCrop = mode ~= 'tone'
	local catalog = LrApplication.activeCatalog()
	local selected = catalog:getTargetPhotos()

	local meta = catalog:batchGetRawMetadata(selected, {
		'path', 'fileFormat', 'colorNameForLabel', 'isoSpeedRating', 'shutterSpeed', 'aperture',
		'focalLength', 'dateTimeOriginal',
	})
	local flashText = catalog:batchGetFormattedMetadata(selected, { 'flash' })

	-- First pass: classify the photos
	local items, nKeys, nSkipped, nChangedByHand = {}, 0, 0, 0
	for _, photo in ipairs(selected) do
		local m = meta[photo]
		if RAW_FORMATS[m.fileFormat] then
			local develop = photo:getDevelopSettings()
			local item = { photo = photo, meta = m, develop = develop, key = m.colorNameForLabel == KEY_LABEL }
			if item.key then
				nKeys = nKeys + 1
			elseif mode == 'redo' then
				item.state = recordState(photo:getPropertyForPlugin(_PLUGIN, 'autoEditRecord'), develop)
				if item.state and item.state.toneChanged then nChangedByHand = nChangedByHand + 1 end
			end
			items[#items + 1] = item
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
	if #items <= nKeys then
		LrDialogs.message('Auto Edit', 'Select the other photos of the event as well as the green-labelled key photos.', 'info')
		return
	end
	if not LrFileUtils.exists(PYTHON) then
		LrDialogs.message('Auto Edit: Python not found', 'Expected the project environment at\n' .. PYTHON, 'critical')
		return
	end

	-- Redo: decide what to do with photos changed by hand since their last auto edit
	local redoChanged = false
	if mode == 'redo' and nChangedByHand > 0 then
		local answer = LrDialogs.confirm(
			string.format('%d photo(s) were changed since their last auto edit', nChangedByHand),
			'Their tone settings have hand edits, a synced or pasted look, a history step or a reset since the plugin last edited them.\n\n'
				.. 'Keep them as they are, or redo them too (replacing those changes)?',
			'Keep my changes', 'Cancel', 'Redo them too')
		if answer == 'cancel' then return end
		redoChanged = answer == 'other'
	end

	-- Build the request; remember which photos must be left alone
	local records, byId, currentById, keep, nCropped, pathById = {}, {}, {}, {}, 0, {}
	for _, item in ipairs(items) do
		local photo, m = item.photo, item.meta
		local id = photo.localIdentifier
		local cropped = isCropped(item.develop)
		local st = item.state
		if not item.key and mode == 'redo' and st then
			if st.toneChanged and not redoChanged then
				keep[id] = true -- hand edits: leave the whole photo alone
			elseif not st.cropChanged or (st.toneChanged and redoChanged) then
				cropped = false -- recompute the crop; a hand-only crop change is kept, without asking
			end
		end
		if not item.key and cropped and not keep[id] then nCropped = nCropped + 1 end
		local flash = (flashText[photo].flash or ''):lower()
		local rec = {
			id = id, path = m.path, key = item.key, cropped = cropped,
			iso = m.isoSpeedRating, shutter = m.shutterSpeed, aperture = m.aperture,
			focal = m.focalLength, time = m.dateTimeOriginal,
			flash = flash:find('fire') ~= nil and flash:find('not') == nil,
		}
		records[#records + 1] = jsonObject(rec, item.key and { settings = jsonObject(item.develop) } or nil)
		byId[id] = photo
		currentById[id] = item.develop
		pathById[id] = m.path
	end

	local tmp = LrPathUtils.getStandardFilePath('temp')
	local reqPath = LrPathUtils.child(tmp, 'autoedit_request.json')
	local respPath = LrPathUtils.child(tmp, 'autoedit_response.tsv')
	local logPath = LrPathUtils.child(tmp, 'autoedit_log.txt')
	local progressPath = LrPathUtils.child(tmp, 'autoedit_progress.txt')
	LrFileUtils.delete(respPath)
	LrFileUtils.delete(respPath .. '.skipped')
	LrFileUtils.delete(progressPath)

	local f = io.open(reqPath, 'w')
	f:write('{"photos":[' .. table.concat(records, ',\n') .. ']}')
	f:close()

	local progress = LrProgressScope { title = 'Auto Edit: ' .. #items .. ' photos', functionContext = context }
	progress:setCaption('Starting the models...')
	-- On Windows LrTasks.execute needs the whole command wrapped in an extra pair of quotes
	local cmd = '"' .. '"' .. PYTHON .. '" "' .. SCRIPT .. '" "' .. reqPath .. '" "' .. respPath .. '"'
		.. (withCrop and '' or ' --no-crop') .. ' --progress "' .. progressPath .. '"'
		.. ' > "' .. logPath .. '" 2>&1' .. '"'

	-- Run Python in its own task and poll its progress file, so Lightroom's activity bar moves.
	local status, finished = nil, false
	LrTasks.startAsyncTask(function()
		status = LrTasks.execute(cmd)
		finished = true
	end)
	while not finished do
		LrTasks.sleep(0.5)
		local stage, done, total = readFile(progressPath):match('^([^\t]+)\t(%d+)\t(%d+)')
		if stage then
			done, total = tonumber(done), tonumber(total)
			progress:setCaption(string.format('%s: %d of %d', stage, done, total))
			if total > 0 then progress:setPortionComplete(done, total) end
		end
	end
	progress:done()

	if status ~= 0 or not LrFileUtils.exists(respPath) then
		LrDialogs.message('Auto Edit failed', tail(readFile(logPath), 1500), 'critical')
		return
	end

	local settings, count, crops = parseResponse(respPath)
	local skippedNames = {}
	for _, id in ipairs(readSkipped(respPath .. '.skipped')) do
		skippedNames[#skippedNames + 1] = pathById[id] and LrPathUtils.leafName(pathById[id]) or tostring(id)
	end
	local skippedText = ''
	if #skippedNames > 0 then
		skippedText = string.format('\n\n%d photo(s) could not be read (damaged or missing files) and were left alone: %s',
			#skippedNames, nameList(skippedNames, 12))
	end
	-- Only apply to photos that may be changed and whose settings would actually change
	local changed, nChanged, nNewCrops = {}, 0, 0
	for id, s in pairs(settings) do
		if not keep[id] and currentById[id] and differs(currentById[id], s) then
			changed[id] = s
			nChanged = nChanged + 1
			if s.CropLeft ~= nil then nNewCrops = nNewCrops + 1 end
		end
	end
	local nKept = 0
	for _ in pairs(keep) do nKept = nKept + 1 end

	if nChanged == 0 then
		LrDialogs.message('Auto Edit: nothing to change',
			string.format('All %d photo(s) already have these settings, so there is nothing to apply.', count - nKept)
				.. skippedText, 'info')
		return
	end

	local info = string.format('%d key photo(s) -> %d photo(s) will be updated.', nKeys, nChanged)
	if count - nKept > nChanged then
		info = info .. string.format('\n%d already have these settings and are skipped.', count - nKept - nChanged)
	end
	if nKept > 0 then
		info = info .. string.format('\n%d photo(s) you changed by hand are kept as they are.', nKept)
	end
	if withCrop then
		info = info .. string.format('\n%d get a new crop (straightened where the camera was tilted).', nNewCrops)
		if nCropped > 0 then
			info = info .. string.format('\n%d already-cropped photo(s) keep their crop.', nCropped)
		end
	end
	if nSkipped > 0 then
		info = info .. string.format('\n%d non-raw file(s) in the selection are left alone.', nSkipped)
	end
	info = info .. '\nKey photos are not changed. Undo with Ctrl+Z.' .. skippedText
	if LrDialogs.confirm('Apply ' .. MODE_TITLES[mode] .. '?', info, 'Apply', 'Cancel') ~= 'ok' then
		return
	end

	local applied = 0
	local stamp = os.date('%Y-%m-%d %H:%M')
	catalog:withWriteAccessDo(MODE_TITLES[mode], function()
		for id, s in pairs(changed) do
			local photo = byId[id]
			if photo then
				photo:applyDevelopSettings(s)
				-- What the photo now holds for every setting the plugin manages: new values over old ones
				local record = parseRecord(photo:getPropertyForPlugin(_PLUGIN, 'autoEditRecord'))
				if s.CropLeft ~= nil then
					for _, k in ipairs(CROP_KEYS) do record[k] = nil end
				end
				for k, v in pairs(s) do record[k] = v end
				photo:setPropertyForPlugin(_PLUGIN, 'autoEditRecord', serializeRecord(record))
				photo:setPropertyForPlugin(_PLUGIN, 'autoEditDate', stamp)
				photo:setPropertyForPlugin(_PLUGIN, 'autoEditMode', MODE_TITLES[mode])
				applied = applied + 1
			end
		end
	end, { timeout = 120 })

	local doneText = string.format('%s applied to %d photo(s)', MODE_TITLES[mode], applied)
	if LrDialogs.showBezel then
		LrDialogs.showBezel(doneText)
	else
		LrDialogs.message('Auto Edit finished', doneText .. '. Undo with Ctrl+Z.', 'info')
	end
end

local function start(mode)
	LrTasks.startAsyncTask(function()
		LrFunctionContext.callWithContext('AutoEdit', function(context)
			LrDialogs.attachErrorDialogToFunctionContext(context)
			run(context, mode)
		end)
	end)
end

return {
	start = start, jsonObject = jsonObject, parseResponse = parseResponse, isCropped = isCropped,
	differs = differs, serializeRecord = serializeRecord, parseRecord = parseRecord, recordState = recordState,
}
