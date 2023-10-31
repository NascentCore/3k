import Vue from 'vue'
import VueI18n from 'vue-i18n'
import enLocale from './en'
import zhLocale from './zh'
import locale from 'element-ui/lib/locale'
import elementEnLocale from 'element-ui/lib/locale/lang/en'
import elementZhLocale from 'element-ui/lib/locale/lang/zh-CN'

Vue.use(VueI18n)
const messages = {
  en: {
    ...enLocale,
    ...elementEnLocale
  },
  zh: {
    ...zhLocale,
    ...elementZhLocale
  }
}

const getDefaultLang = () => {
  const localLang = navigator.language
  const lang = localLang || 'enUS'
  localStorage.setItem('lang', localLang)
  if (lang === 'zh' || lang === 'zh-CN') {
    return 'zh'
  } else {
    return 'en'
  }
}
const i18n = new VueI18n({
  silentTranslationWarn: true,
  locale: getDefaultLang(),
  fallbackLocale: 'zh',
  messages
})
locale.i18n((key, value) => i18n.t(key, value))
export default i18n
