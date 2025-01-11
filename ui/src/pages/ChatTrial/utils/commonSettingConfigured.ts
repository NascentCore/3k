import { SubmitKey } from './interface';

const Key = 'CommonSettingConfigured';

const initCommonSettingConfigured = { submitKey: SubmitKey.Enter, webSiteTitle: '算想未来' };

export const saveCommonSettingConfigured = (data: any) => {
  localStorage.setItem(Key, JSON.stringify(data));
};

export const getCommonSettingConfigured = (): any => {
  const _settingConfig = localStorage.getItem(Key);
  if (_settingConfig) {
    const _settingConfigJson = JSON.parse(_settingConfig);
    return _settingConfigJson;
  } else {
    saveCommonSettingConfigured(initCommonSettingConfigured);
    return getCommonSettingConfigured();
  }
};

export const resetCommonSettingConfigured = () => {
  saveCommonSettingConfigured(initCommonSettingConfigured);
};
