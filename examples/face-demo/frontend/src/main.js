import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL
if (!baseURL) {
  console.warn('VITE_API_BASE_URL 未设置')
}
axios.defaults.baseURL = baseURL

const app = createApp(App)
app.use(ElementPlus)
app.use(router)
app.mount('#app') 