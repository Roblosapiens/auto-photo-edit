return {
	LrSdkVersion = 7.0,
	LrSdkMinimumVersion = 6.0,
	LrToolkitIdentifier = 'com.roblosapiens.autophotoedit', -- never change: catalog fields are tied to it
	LrPluginName = 'Auto Photo Edit',
	LrPluginInfoUrl = 'https://github.com/Roblosapiens/auto-photo-edit',

	LrLibraryMenuItems = {
		{ title = 'Auto Edit + Crop from Key Photos (green label)', file = 'AutoEdit.lua' },
		{ title = 'Auto Edit + Crop: redo from key photos (green label)', file = 'AutoEditRedo.lua' },
		{ title = 'Auto Edit from Key Photos, tone only (green label)', file = 'AutoEditToneOnly.lua' },
	},

	LrMetadataProvider = 'MetadataDefinition.lua',
	LrMetadataTagsetFactory = 'Tagset.lua',

	VERSION = { major = 0, minor = 3, revision = 0 },
}
