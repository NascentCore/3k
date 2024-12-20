import { createRouter, createWebHistory } from 'vue-router'
import ImageManagement from './views/ImageManagement.vue'
import Search from './views/Search.vue'

const routes = [
  { 
    path: '/', 
    redirect: '/image-management'
  },
  {
    path: '/image-management',
    name: 'ImageManagement',
    component: ImageManagement
  },
  {
    path: '/search',
    name: 'Search',
    component: Search
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router 