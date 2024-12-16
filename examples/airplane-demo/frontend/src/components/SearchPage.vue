<template>
  <div class="search-page">
    <div class="search-container">
      <div class="logo-container">
        <img :src="logoUrl" alt="Logo" class="logo">
      </div>
      <div class="search-box">
        <el-input
          v-model="searchForm.keyword"
          placeholder="搜索航空图片..."
          class="search-input"
          @keyup.enter="handleSearch"
        >
          <template #suffix>
            <input
              type="file"
              accept="image/*"
              @change="handleFileUpload"
              style="display: none"
              ref="fileInput"
            >
            <el-button 
              class="upload-button" 
              type="primary" 
              plain
              @click="$refs.fileInput.click()"
            >
              <el-icon><Camera /></el-icon>
            </el-button>
          </template>
          <template #append>
            <el-button type="primary" @click="handleSearch">搜索</el-button>
            <div class="model-switch-container">
              <el-switch
                v-model="searchForm.use_optimized_model"
                class="model-switch"
                active-text="模型优化"
              />
              <el-tooltip
                content="开启模型优化后，会使用大模型优化查询语句，检索结果更精准，查询速度会变慢"
                placement="top"
                effect="light"
                :popper-style="{maxWidth: '200px', wordWrap: 'break-word', whiteSpace: 'pre-wrap'}"
              >
                <el-icon class="help-icon"><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
          </template>
        </el-input>
        
        <div class="search-examples">
          <span class="example-label">示例搜索：</span>
          <a 
            v-for="(example, index) in searchExamples" 
            :key="index" 
            @click="executeExampleSearch(example)"
            class="example-link"
          >
            {{ example }}
          </a>
        </div>
      </div>
    </div>

    <div class="search-results">
      <div v-if="uploadedImageUrl || (hasSearched && searchImageUrl)" class="search-reference">
        <h3 class="reference-title">您上传的图片：</h3>
        <div class="reference-image-container">
          <img 
            :src="uploadedImageUrl || searchImageUrl" 
            class="reference-image" 
            @click="showLargeImage(uploadedImageUrl || searchImageUrl)"
          >
          <el-icon 
            v-if="uploadedImageUrl" 
            class="remove-reference-image" 
            @click="removeThumbnail"
          >
            <Close />
          </el-icon>
        </div>
      </div>

      <div v-if="isSearching" class="loading-container">
        <el-skeleton :rows="4" animated />
        <el-skeleton :rows="4" animated />
        <el-skeleton :rows="4" animated />
      </div>
      
      <div v-else-if="hasSearched && searchResults.length === 0" class="no-results">
        暂无搜索结果
      </div>
      
      <template v-else-if="searchResults.length > 0">
        <h3 v-if="searchImageUrl" class="similar-images-title">相似图片</h3>
        <el-row :gutter="20">
          <el-col 
            v-for="item in searchResults" 
            :key="item.id" 
            :xs="24" 
            :sm="12" 
            :md="8" 
            :lg="6"
          >
            <el-card class="image-card">
              <img 
                :src="item.image_url" 
                class="image" 
                @click="showLargeImage(item.image_url)"
              >
              <div class="image-info">
                <p>航空公司: {{ item.airline }}</p>
                <p>机型: {{ item.aircraft_type }}</p>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </template>
    </div>

    <!-- 添加图片预览对话框 -->
    <el-dialog
      v-model="dialogVisible"
      :show-close="true"
      :modal="true"
      width="80%"
      class="image-dialog"
    >
      <img :src="currentImage" class="large-image" />
    </el-dialog>
  </div>
</template>

<script>
import { ref } from 'vue'
import axios from 'axios'
import { Camera, Upload, Search, QuestionFilled, Close } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import logoUrl from '../assets/logo.png'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL

export default {
  name: 'SearchPage',
  components: {
    Camera,
    Upload,
    Search,
    QuestionFilled,
    Close
  },
  setup() {
    const searchForm = ref({
      keyword: '',
      use_optimized_model: true
    })
    
    const searchResults = ref([])

    const dialogVisible = ref(false)
    const currentImage = ref('')
    const uploadedImageUrl = ref('')
    const fileInput = ref(null)
    const hasSearched = ref(false)
    const isSearching = ref(false)
    const searchImageUrl = ref('')

    const searchExamples = [
      "东方航空C919",
      "汉莎航空",
      "波音777客机",
    ]
    
    const executeExampleSearch = (example) => {
      searchForm.value.keyword = example
      handleSearch()
    }

    const handleSearch = async () => {
      if (isSearching.value) {
        return
      }
      
      try {
        isSearching.value = true
        hasSearched.value = true
        const params = {
          keyword: searchForm.value.keyword,
          use_optimized_model: searchForm.value.use_optimized_model
        }
        
        if (uploadedImageUrl.value) {
          params.image_url = uploadedImageUrl.value
          searchImageUrl.value = uploadedImageUrl.value
        } else {
          searchImageUrl.value = ''
        }
        
        const response = await axios.get(`${API_BASE_URL}/search`, { params })
        console.log('搜索结果:', response.data)
        searchResults.value = response.data.data
        
        if (uploadedImageUrl.value) {
          uploadedImageUrl.value = ''
          searchForm.value.keyword = ''
        }
      } catch (error) {
        console.error('搜索失败:', error)
        ElMessage.error('搜索失败，请稍后重试')
      } finally {
        isSearching.value = false
      }
    }

    const beforeUpload = (file) => {
      const isImage = file.type.startsWith('image/')
      if (!isImage) {
        ElMessage.error('只能上传图片文件！')
        return false
      }
      return true
    }

    const handleUploadSuccess = (response) => {
      if (response.success) {
        ElMessage.success('图片上传成功')
        uploadedImageUrl.value = response.data.url
      } else {
        ElMessage.error('图片上传失败')
      }
    }

    const showLargeImage = (imageUrl) => {
      currentImage.value = imageUrl
      dialogVisible.value = true
    }

    const handleFileUpload = async (event) => {
      const file = event.target.files[0]
      if (!file) return
      
      console.log('开始上传文件:', file.name)
      
      const formData = new FormData()
      formData.append('image', file)
      
      try {
        console.log('发送上传请求...')
        const response = await axios.post(
          `${API_BASE_URL}/upload`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }
        )
        
        console.log('上传响应:', response)
        
        if (response.data && response.data.url) {
          ElMessage.success('图片上传成功')
          uploadedImageUrl.value = response.data.url
          searchResults.value = []
          hasSearched.value = false
          searchImageUrl.value = ''
          handleSearch()
        } else {
          ElMessage.error('图片上传失败：服务器响应格式错误')
        }
      } catch (error) {
        console.error('上传失败:', error)
        console.error('错误详情:', error.response?.data || error.message)
        ElMessage.error(`上传失败：${error.response?.data?.message || error.message}`)
      }
      
      event.target.value = ''
    }

    const removeThumbnail = () => {
      uploadedImageUrl.value = ''
    }

    return {
      searchForm,
      searchResults,
      handleSearch,
      Camera,
      beforeUpload,
      handleUploadSuccess,
      logoUrl,
      dialogVisible,
      currentImage,
      showLargeImage,
      fileInput,
      handleFileUpload,
      uploadedImageUrl,
      hasSearched,
      removeThumbnail,
      isSearching,
      searchImageUrl,
      searchExamples,
      executeExampleSearch
    }
  }
}
</script>

<style scoped>
.search-page {
  min-height: 100vh;
  background-color: #fff;
  color: #333;
  padding: 0;
  margin: 0;
}

.search-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 40vh;
  padding: 20px;
}

.logo {
  margin-bottom: 30px;
}

.logo-img {
  width: 270px;
  height: auto;
}

.search-box {
  width: 100%;
  max-width: 650px;
  margin: 0 auto;
  position: relative;
}

.search-input {
  width: 100%;
}

:deep(.el-input__wrapper) {
  background-color: #fff;
  border: 1px solid #dcdfe6;
  border-radius: 10px;
  height: 44px;
  padding: 0 10px;
  padding-right: 0;
  box-shadow: none !important;
}

:deep(.el-input__wrapper.is-focus) {
  box-shadow: none !important;
}

:deep(.el-input__inner) {
  color: #000;
  font-size: 16px;
}

:deep(.el-input__inner::placeholder) {
  color: #999;
}

.upload-icon {
  margin-right: 8px;
  margin-left: 8px;
}

.upload-button {
  height: 32px;
  width: 32px;
  padding: 6px;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 8px;
}

:deep(.el-input__suffix) {
  right: 0;
}

:deep(.el-input-group__append) {
  display: flex;
  align-items: center;
  padding: 0 10px;
  gap: 10px;
  background: transparent;
  border: none;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

:deep(.el-input-group__append .el-button) {
  border-radius: 10px;
  margin: 0;
  border: 1px solid #409eff;
  height: 44px;
  padding: 0 25px;
  font-size: 16px;
  background-color: #409eff;
  color: #fff;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

:deep(.el-input-group__append .el-button:hover) {
  background-color: #66b1ff;
  border-color: #66b1ff;
  color: #fff;
}

.image-card {
  margin-bottom: 20px;
  background-color: #fff;
  border: 1px solid #eee;
  color: #333;
}

:deep(.el-card) {
  background-color: #fff;
  border: 1px solid #eee;
  color: #333;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

.image {
  width: 100%;
  height: 200px;
  object-fit: cover;
  cursor: pointer;
  transition: opacity 0.3s;
}

.image:hover {
  opacity: 0.8;
}

.image-info {
  margin-top: 10px;
  padding: 10px;
}

.image-info p {
  margin: 5px 0;
  color: #333;
}

.search-results {
  padding: 20px;
  max-width: 95%;
  margin: 0 auto;
}

:deep(.el-row) {
  margin-left: 0 !important;
  margin-right: 0 !important;
}

:deep(.el-col) {
  padding-left: 10px !important;
  padding-right: 10px !important;
}

.large-image {
  width: 100%;
  height: auto;
  max-height: 80vh;
  object-fit: contain;
}

:deep(.image-dialog .el-dialog) {
  display: flex;
  flex-direction: column;
  margin: 0 auto;
  max-width: 1200px;
  background-color: transparent;
  box-shadow: none;
}

:deep(.image-dialog .el-dialog__body) {
  padding: 0;
  background-color: transparent;
  display: flex;
  justify-content: center;
  align-items: center;
}

.model-switch {
  margin-left: 15px;
}

:deep(.el-switch__label) {
  color: #333;
}

.no-results {
  text-align: center;
  padding: 40px 0;
  color: #909399;
  font-size: 16px;
}

.model-switch-container {
  display: flex;
  align-items: center;
  gap: 5px;
}

.help-icon {
  color: #909399;
  font-size: 16px;
  cursor: pointer;
}

.help-icon:hover {
  color: #409eff;
}

.remove-reference-image {
  position: absolute;
  top: -8px;
  right: -8px;
  background-color: #fff;
  border-radius: 50%;
  padding: 2px;
  font-size: 14px;
  cursor: pointer;
  color: #909399;
  border: 1px solid #dcdfe6;
}

.remove-reference-image:hover {
  color: #f56c6c;
  border-color: #f56c6c;
}

.loading-container {
  padding: 20px;
}

:deep(.el-skeleton) {
  padding: 20px;
  margin-bottom: 20px;
}

.search-reference {
  margin: 20px 0;
  text-align: left;
}

.reference-title {
  font-size: 16px;
  color: #606266;
  margin-bottom: 15px;
  padding: 0;
}

.reference-image-container {
  display: inline-block;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  padding: 10px;
  background-color: #fff;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

.reference-image {
  max-width: 300px;
  max-height: 200px;
  object-fit: contain;
  cursor: pointer;
  transition: opacity 0.3s;
}

.reference-image:hover {
  opacity: 0.8;
}

:deep(.el-tooltip__popper) {
  max-width: 300px;
  line-height: 1.5;
  word-wrap: break-word;
  white-space: pre-wrap;
}

:deep(.el-popper) {
  max-width: 300px !important;
}

:deep(.el-popper__content) {
  line-height: 1.5;
  word-wrap: break-word;
  white-space: pre-wrap !important;
}

.search-examples {
  margin-top: 15px;
  font-size: 14px;
  color: #606266;
  text-align: left;
  width: 100%;
}

.example-label {
  margin-right: 10px;
}

.example-link {
  color: #409EFF;
  margin: 0 10px;
  cursor: pointer;
  text-decoration: none;
}

.example-link:hover {
  color: #66b1ff;
  text-decoration: underline;
}

.similar-images-title {
  font-size: 16px;
  color: #606266;
  margin: 20px 0;
  padding: 0;
  text-align: left;
}
</style> 