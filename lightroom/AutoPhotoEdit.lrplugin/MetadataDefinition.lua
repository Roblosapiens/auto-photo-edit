--[[
Catalog fields the plugin writes on every photo it edits. They live only in the catalog (never in the
image files). "Auto edited" and "Auto edit mode" can be shown in the Metadata panel (tagset
"Auto Photo Edit") and used in the Library Filter bar; autoEditRecord is internal: the exact values
applied, used by the redo mode to tell untouched auto edits from later hand edits.

Bump schemaVersion whenever these fields change; Lightroom then updates the catalog.
]]

return {
	metadataFieldsForPhotos = {
		{
			id = 'autoEditDate',
			title = 'Auto edited',
			dataType = 'string',
			readOnly = true,
			searchable = true,
			browsable = true,
		},
		{
			id = 'autoEditMode',
			title = 'Auto edit mode',
			dataType = 'string',
			readOnly = true,
			searchable = true,
			browsable = true,
		},
		{
			id = 'autoEditRecord', -- no title: hidden from the Metadata panel
			dataType = 'string',
		},
	},
	schemaVersion = 1,
}
