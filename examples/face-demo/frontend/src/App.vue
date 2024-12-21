<template>
  <div id="app">
    <el-header>
      <div class="header-content">
        <img src="@/assets/logo.png" class="logo" alt="logo">
        <h1>人像图库</h1>
      </div>
    </el-header>
    <el-container class="main-container">
      <el-aside :width="isCollapse ? '64px' : '200px'">
        <div class="toggle-button" @click="toggleCollapse">
          <el-icon><Fold v-if="!isCollapse"/><Expand v-else/></el-icon>
        </div>
        <el-menu
          :default-active="activeMenu"
          class="el-menu-vertical"
          :collapse="isCollapse"
          background-color="#f0f2f5"
          text-color="#303133"
          active-text-color="#1890ff"
          router>
          <el-menu-item index="/image-management">
            <el-icon><Picture /></el-icon>
            <template #title>图片管理</template>
          </el-menu-item>
          <el-menu-item index="/search">
            <el-icon><Search /></el-icon>
            <template #title>图片搜索</template>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { 
  ElContainer, 
  ElAside, 
  ElMenu, 
  ElMenuItem,
  ElHeader,
  ElMain, 
  ElRow, 
  ElCol,
  ElIcon
} from 'element-plus'
import { 
  Picture,
  Search,
  Fold,
  Expand
} from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'

export default {
  name: 'App',
  components: {
    ElContainer,
    ElAside,
    ElMenu,
    ElMenuItem,
    ElHeader,
    ElMain,
    ElRow,
    ElCol,
    ElIcon,
    Picture,
    Search,
    Fold,
    Expand
  },
  setup() {
    const route = useRoute()
    const isCollapse = ref(false)
    
    const activeMenu = computed(() => route.path)

    const toggleCollapse = () => {
      isCollapse.value = !isCollapse.value
    }

    return {
      isCollapse,
      toggleCollapse,
      activeMenu
    }
  }
}
</script>

<style scoped>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-container {
  flex: 1;
  height: calc(100vh - 64px);
}

.el-aside {
  background-color: #f0f2f5;
  transition: width 0.3s;
  height: 100%;
  border-right: 1px solid #e4e7ed;
}

.el-menu-vertical:not(.el-menu--collapse) {
  width: 200px;
}

.el-menu {
  border-right: none;
}

.toggle-button {
  height: 40px;
  color: #606266;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border-bottom: 1px solid #e4e7ed;
}

.toggle-button:hover {
  background-color: #e6f7ff;
}

.el-header {
  text-align: left;
  padding: 0 20px;
  background-color: #fff;
  color: #303133;
  height: 64px !important;
  line-height: 64px;
  border-bottom: 1px solid #e4e7ed;
}

h1 {
  margin: 0;
  font-size: 24px;
  font-weight: 500;
}

.el-main {
  padding: 20px;
  background-color: #fff;
}

h2 {
  color: #303133;
  margin-bottom: 20px;
  font-weight: 500;
}

p {
  font-size: 16px;
  line-height: 1.6;
  color: #606266;
}

.el-menu-item:hover {
  background-color: #e6f7ff !important;
}

.el-menu-item.is-active {
  background-color: #e6f7ff !important;
  color: #1890ff !important;
}

.el-menu-item.is-active .el-icon {
  color: #1890ff !important;
}

.header-content {
  display: flex;
  align-items: center;
  height: 100%;
}

.logo {
  height: 32px;
  margin-right: 12px;
}
</style> 