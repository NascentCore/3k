<template>
  <div class="search-container">
    <h2>图片搜索</h2>
    
    <div class="text-search-area">
      <el-input
        v-model="searchText"
        placeholder="请输入搜索关键词"
        class="text-search-input"
        @keyup.enter="handleTextSearch"
      >
        <template #append>
          <el-button type="primary" @click="handleTextSearch" :loading="textSearching" style="background-color: var(--el-color-primary); color: white;">
            搜索
          </el-button>
        </template>
      </el-input>
    </div>

    <div class="search-area" v-loading="searching" 
         element-loading-text="正在搜索中..."
         element-loading-background="rgba(255, 255, 255, 0.8)">
      <el-upload
        class="upload-demo"
        drag
        action="#"
        :auto-upload="false"
        :on-change="handleFileChange"
        :on-exceed="handleExceed"
        :limit="1"
        accept="image/*"
        ref="uploadRef"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽图片到此处或 <em>点击上传</em>
          <div class="upload-tip">支持粘贴图片 (Ctrl+V / Command+V)</div>
        </div>
      </el-upload>
    </div>

    <div v-if="uploadedImageUrl" class="uploaded-image-section">
      <h3>您上传的图片</h3>
      <el-card class="uploaded-image-card">
        <img :src="uploadedImageUrl" class="uploaded-image" @click="previewUploadedImage"/>
      </el-card>
    </div>

    <div class="search-results" v-if="searchResults.length">
      <h3>搜索结果</h3>
      <el-card class="result-list">
        <el-row :gutter="20">
          <el-col v-for="(item, index) in searchResults" 
                  :key="index" 
                  :xs="24" 
                  :sm="12" 
                  :md="8" 
                  :lg="6" 
                  :xl="4">
            <el-card class="result-card" :body-style="{ padding: '0px' }">
              <div class="image-container">
                <img :src="item.url" 
                     class="result-image"
                     @click="handlePreview(item)"
                     @load="(e) => handleImageLoad(e, index)" />
                <div v-for="(face, faceIndex) in item.faces" 
                     :key="faceIndex"
                     class="face-box"
                     :style="getFaceBoxStyle(face, index)">
                </div>
              </div>
              <div class="result-info">
                <span v-if="!isNaN(item.similarity)">相似度: {{ (item.similarity * 100).toFixed(2) }}%</span>
                <div class="result-actions">
                  <el-button type="text" @click="handleDownload(item)">
                    <el-icon><Download /></el-icon>
                  </el-button>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-card>

      <el-image-viewer
        v-if="showViewer"
        @close="showViewer = false"
        :url-list="[previewUrl]"
        :initial-index="0"
      />
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, Download } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElImageViewer } from 'element-plus'

export default {
  name: 'Search',
  components: {
    UploadFilled,
    Download,
    ElImageViewer
  },
  setup() {
    const selectedImage = ref(null)
    const searching = ref(false)
    const searchResults = ref([])
    const uploadRef = ref(null)
    const showViewer = ref(false)
    const previewUrl = ref('')
    const uploadedImageUrl = ref('')
    const imageElements = ref([])
    const searchText = ref('')
    const textSearching = ref(false)

    const handleExceed = () => {
      ElMessage({
        message: '只能上传一张图片',
        type: 'warning',
        duration: 2000,
        showClose: true
      })
    }

    const handleFileChange = async (file, fileList) => {
      if (fileList.length > 1) {
        fileList.splice(0, 1)
      }
      selectedImage.value = file.raw
      
      uploadedImageUrl.value = URL.createObjectURL(file.raw)
      
      searching.value = true
      try {
        const formData = new FormData()
        formData.append('file', file.raw)

        const response = await axios.post('/api/image/search', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        if (response.data.status === 'success') {
          searchResults.value = response.data.data
            .map(item => ({
              url: item.oss_url,
              filename: item.original_filename,
              similarity: Math.max(...item.faces.map(face => face.similarity)),
              faces: item.faces
            }))
            .sort((a, b) => b.similarity - a.similarity)
        } else {
          ElMessage.error(response.data.message || '搜索失败')
        }
      } catch (error) {
        console.error('搜索错误:', error)
        ElMessage.error('搜索失败: ' + (error.response?.data?.message || '未知错误'))
      } finally {
        searching.value = false
        uploadRef.value.clearFiles()
      }
      imageElements.value = new Array(searchResults.value.length)
    }

    const handlePreview = (item) => {
      previewUrl.value = item.url
      showViewer.value = true
    }

    const handleDownload = (item) => {
      const link = document.createElement('a')
      link.href = item.url
      link.download = item.filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }

    const previewUploadedImage = () => {
      previewUrl.value = uploadedImageUrl.value
      showViewer.value = true
    }

    const handlePaste = async (event) => {
      const items = event.clipboardData?.items
      if (!items) return

      for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
          const file = item.getAsFile()
          if (file) {
            const uploadFile = {
              raw: file,
              name: `pasted-image-${Date.now()}.png`
            }
            await handleFileChange(uploadFile, [uploadFile])
            break
          }
        }
      }
    }

    const handleImageLoad = (event, index) => {
      imageElements.value[index] = {
        width: event.target.naturalWidth,
        height: event.target.naturalHeight
      }
    }

    const getFaceBoxStyle = (face, imageIndex) => {
      const imageSize = imageElements.value[imageIndex]
      if (!imageSize) return {}
      
      const containerWidth = 200  // 容器宽度
      const containerHeight = 200 // 容器高度
      
      // 计算图片在容器中的实际显示尺寸
      const imageRatio = imageSize.width / imageSize.height
      const containerRatio = containerWidth / containerHeight
      
      let displayWidth, displayHeight, offsetX = 0, offsetY = 0
      
      if (imageRatio > containerRatio) {
        displayHeight = containerHeight
        displayWidth = imageSize.width * (containerHeight / imageSize.height)
        offsetX = (containerWidth - displayWidth) / 2
      } else {
        displayWidth = containerWidth
        displayHeight = imageSize.height * (containerWidth / imageSize.width)
        offsetY = (containerHeight - displayHeight) / 2
      }
      
      // 计算人脸框在显示时的实际尺寸
      const boxWidth = ((face.face_coords.x2 - face.face_coords.x1) / imageSize.width) * displayWidth
      const boxHeight = ((face.face_coords.y2 - face.face_coords.y1) / imageSize.height) * displayHeight
      
      // 根据人脸框大小动态计算框粗细
      // 当框小于 30px 时使用 1.5px，小于 60px 时使用 2px，其他情况使用 3px
      const borderWidth = Math.min(boxWidth, boxHeight) < 30 ? 1.5 : 
                         Math.min(boxWidth, boxHeight) < 60 ? 2 : 3
      
      return {
        left: `${offsetX + (face.face_coords.x1 / imageSize.width) * displayWidth}px`,
        top: `${offsetY + (face.face_coords.y1 / imageSize.height) * displayHeight}px`,
        width: `${boxWidth}px`,
        height: `${boxHeight}px`,
        borderWidth: `${borderWidth}px`  // 动态设置边框粗细
      }
    }

    const handleTextSearch = async () => {
      if (!searchText.value.trim()) {
        ElMessage({
          message: '请输入搜索关键词',
          type: 'warning',
          duration: 2000,
          showClose: true
        })
        return
      }

      textSearching.value = true
      try {
        const response = await axios.post('/api/text/search', {
          query: searchText.value.trim()
        })
        
        if (response.data.status === 'success') {
          searchResults.value = response.data.data
            .map(item => ({
              url: item.oss_url,
              filename: item.original_filename,
              similarity: Math.max(...item.faces.map(face => face.similarity)),
              faces: item.faces
            }))
            .sort((a, b) => b.similarity - a.similarity)
          imageElements.value = new Array(searchResults.value.length)
        } else {
          ElMessage.error(response.data.message || '搜索失败')
        }
      } catch (error) {
        console.error('搜索错误:', error)
        ElMessage.error('搜索失败: ' + (error.response?.data?.message || '未知错误'))
      } finally {
        textSearching.value = false
      }
    }

    onMounted(() => {
      document.addEventListener('paste', handlePaste)
    })

    onUnmounted(() => {
      document.removeEventListener('paste', handlePaste)
    })

    return {
      uploadRef,
      selectedImage,
      searching,
      searchResults,
      handleFileChange,
      handleExceed,
      showViewer,
      previewUrl,
      handlePreview,
      handleDownload,
      uploadedImageUrl,
      previewUploadedImage,
      imageElements,
      handleImageLoad,
      getFaceBoxStyle,
      searchText,
      textSearching,
      handleTextSearch,
    }
  }
}
</script>

<style scoped>
.search-container {
  padding: 20px;
}

.text-search-area {
  max-width: 500px;
  margin: 20px auto;
}

.text-search-input {
  width: 100%;
}

.search-area {
  max-width: 500px;
  margin: 20px auto;
  min-height: 200px;
  position: relative;
}

.search-results {
  margin-top: 40px;
}

.result-list {
  min-height: 400px;
}

.result-card {
  margin-bottom: 20px;
  transition: all 0.3s;
}

.result-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 2px 12px 0 rgba(0,0,0,.1);
}

.result-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  cursor: pointer;
}

.result-image:hover {
  opacity: 0.8;
}

.result-info {
  padding: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-actions {
  display: flex;
  gap: 8px;
}

.result-actions .el-button {
  padding: 2px;
}

.el-upload__tip {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
}

.uploaded-image-section {
  margin: 40px 0;
  text-align: left;
}

.uploaded-image-card {
  margin-top: 20px;
  text-align: left;
  max-width: 300px;
}

.uploaded-image {
  width: 100%;
  height: 200px;
  object-fit: contain;
  cursor: pointer;
  margin: 0;
}

.uploaded-image:hover {
  opacity: 0.8;
}

.upload-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
}

.image-container {
  position: relative;
  width: 200px;  /* 固定容器尺寸 */
  height: 200px;
  margin: 0 auto;  /* 居中显示 */
  overflow: hidden;  /* 防止图片溢出 */
}

.face-box {
  position: absolute;
  border: 2px solid #67C23A;
  box-sizing: border-box;
  pointer-events: none;
}
</style> 