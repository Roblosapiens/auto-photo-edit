-- Metadata panel tagset "Auto Photo Edit": file name plus the plugin's visible fields.
-- Field ids are '<LrToolkitIdentifier>.<field id>' (see Info.lua and MetadataDefinition.lua).
return {
	id = 'autoPhotoEditTagset',
	title = 'Auto Photo Edit',
	items = {
		'com.adobe.filename',
		'com.adobe.separator',
		'com.roblosapiens.autophotoedit.autoEditDate',
		'com.roblosapiens.autophotoedit.autoEditMode',
	},
}
