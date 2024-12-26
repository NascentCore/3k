<template>
  <div class="image-management">
    <div class="page-header">
      <h2>图片管理</h2>
      <el-button type="primary" @click="handleUpload">
        <el-icon><Upload /></el-icon>上传图片
      </el-button>
    </div>
    
    <el-card class="image-list">
      <el-empty v-if="!images.length" description="暂无图片" />
      <template v-else>
        <el-row :gutter="20">
          <el-col 
            v-for="image in uploadingImage ? [uploadingImage, ...images] : images" 
            :key="image.id" 
            :xs="24" :sm="12" :md="8" :lg="6" :xl="4"
          >
            <el-card class="image-card" :body-style="{ padding: '0px' }">
              <div class="image-wrapper">
                <img :src="image.oss_url" class="image" @click="handlePreview(image)" />
                <div v-if="image.uploading" class="upload-overlay">
                  <div class="upload-progress">
                    <el-icon class="upload-icon"><Upload /></el-icon>
                    <div>正在上传...</div>
                    <el-progress 
                      :percentage="100" 
                      :duration="1.5"
                      :indeterminate="true"
                      :stroke-width="2"
                      class="progress-bar"
                    />
                  </div>
                </div>
              </div>
              <div class="image-info">
                <span>{{ image.original_filename }}</span>
                <div class="image-actions" v-if="!image.uploading">
                  <el-button type="text" @click="handleTag(image)">
                    <el-icon><PriceTag /></el-icon>
                  </el-button>
                  <el-button type="text" @click="handleDownload(image)">
                    <el-icon><Download /></el-icon>
                  </el-button>
                  <el-button type="text" @click="handleDelete(image)">
                    <el-icon><Delete /></el-icon>
                  </el-button>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
        
        <!-- 添加分页组件 -->
        <div class="pagination-container">
          <el-pagination
            :current-page="currentPage"
            :page-size="pageSize"
            :page-sizes="[12, 24, 36]"
            :total="total"
            layout="total, sizes, prev, pager, next"
            @update:current-page="currentPage = $event"
            @update:page-size="pageSize = $event"
            @size-change="handleSizeChange"
            @current-change="handleCurrentChange"
          />
        </div>
      </template>
    </el-card>
    
    <!-- 图片预览组件 -->
    <el-image-viewer
      v-if="showViewer && !showTagViewer"
      @close="showViewer = false"
      :url-list="[previewUrl]"
      :initial-index="0"
    />
    
    <!-- 标签预览组件 -->
    <image-preview
      v-if="showTagViewer"
      v-model="showTagViewer"
      :image-url="previewUrl"
      :image-id="currentImage.id"
      :faces="currentFaces"
      @update:modelValue="showTagViewer = $event"
      @close="handleTagViewerClose"
    />
    
    <!-- 添加隐藏的文件上传输入框 -->
    <input
      type="file"
      ref="fileInput"
      style="display: none"
      accept="image/*"
      @change="onFileSelected"
    />
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { ElMessage, ElImageViewer, ElMessageBox } from 'element-plus'
import { Delete, Download, Upload, PriceTag } from '@element-plus/icons-vue'
import axios from 'axios'
import ImagePreview from '../components/ImagePreview.vue'

export default {
  name: 'ImageManagement',
  components: {
    Delete,
    Download,
    Upload,
    PriceTag,
    ImagePreview
  },
  setup() {
    const images = ref([])
    const fileInput = ref(null)
    const currentPage = ref(1)
    const pageSize = ref(12)
    const total = ref(0)
    const showViewer = ref(false)
    const previewUrl = ref('')
    const uploadingImage = ref(null)
    const showTagViewer = ref(false)
    const currentFaces = ref([])
    const currentImage = ref(null)

    const fetchImages = async () => {
      try {
        const response = await axios.get('/api/images', {
          params: {
            page: currentPage.value,
            page_size: pageSize.value
          }
        })
        
        if (response.data) {
          images.value = response.data.data
          total.value = response.data.total
        } else {
          ElMessage.error('获取图片列表失败')
        }
      } catch (error) {
        console.error('获取图片列表错误:', error)
        ElMessage.error('获取图片列表失败: ' + (error.response?.data?.message || '未知错误'))
      }
    }

    const handleSizeChange = (val) => {
      pageSize.value = val
      fetchImages()
    }

    const handleCurrentChange = (val) => {
      currentPage.value = val
      fetchImages()
    }

    onMounted(() => {
      fetchImages()
    })

    const handleUpload = () => {
      // 触发隐藏的文件输入框的点击事件
      fileInput.value.click()
    }

    const onFileSelected = async (event) => {
      const file = event.target.files[0]
      if (!file) return

      // 创建临时的预览图片对象
      uploadingImage.value = {
        id: 'uploading',
        oss_url: URL.createObjectURL(file),
        original_filename: file.name,
        uploading: true
      }

      const formData = new FormData()
      formData.append('file', file)

      try {
        const response = await axios.post('/api/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        if (response.data.status === 'success') {
          ElMessage.success('图片上传成功')
          // 重新获取第一页的数据
          currentPage.value = 1
          await fetchImages()
        } else if (response.data.status === 'warning') {
          ElMessage.warning(response.data.message)
          await fetchImages()
        } else {
          ElMessage.error(response.data.message || '上传失败')
        }
      } catch (error) {
        console.error('上传错误:', error)
        ElMessage.error('上传失败: ' + (error.response?.data?.message || '未知错误'))
      } finally {
        // 清除临时预览图片
        URL.revokeObjectURL(uploadingImage.value.oss_url)
        uploadingImage.value = null
      }

      event.target.value = ''
    }

    const handlePreview = (image) => {
      previewUrl.value = image.oss_url
      showViewer.value = true
      showTagViewer.value = false
    }

    const handleView = (image) => {
      handlePreview(image)
    }

    const handleDelete = async (image) => {
      try {
        // 显示确认对话框
        await ElMessageBox.confirm(
          '确定要删除这张图片吗？此操作不可恢复。',
          '警告',
          {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning',
          }
        )

        // 调用删除接口
        const response = await axios.delete(`/api/images/${image.id}`)
        
        if (response.data.status === 'success') {
          ElMessage.success('删除成功')
          // 重新获取当前页的数据
          await fetchImages()
        } else {
          ElMessage.error(response.data.message || '删除失败')
        }
      } catch (error) {
        if (error === 'cancel') {
          return
        }
        console.error('删除错误:', error)
        ElMessage.error('删除失败: ' + (error.response?.data?.message || '未知错误'))
      }
    }

    const handleDownload = (image) => {
      // 创建一个临时的 a 标签来触发下载
      const link = document.createElement('a')
      link.href = image.oss_url
      link.download = image.original_filename // 设置下载文件名
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }

    const handleTag = (image) => {
      previewUrl.value = image.oss_url
      currentFaces.value = image.faces || []
      currentImage.value = image
      showTagViewer.value = true
      showViewer.value = false
    }

    const handleTagViewerClose = () => {
      showTagViewer.value = false
      showViewer.value = false
      previewUrl.value = ''
      currentFaces.value = []
    }

    return {
      images,
      fileInput,
      currentPage,
      pageSize,
      total,
      handleUpload,
      handleView,
      handleDelete,
      onFileSelected,
      handleSizeChange,
      handleCurrentChange,
      showViewer,
      previewUrl,
      handlePreview,
      handleDownload,
      uploadingImage,
      handleTag,
      showTagViewer,
      currentFaces,
      handleTagViewerClose,
      currentImage,
    }
  }
}
</script>

<style scoped>
.image-management {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-header h2 {
  margin: 0;
}

.image-list {
  min-height: 400px;
}

.image-card {
  margin-bottom: 20px;
  transition: all 0.3s;
}

.image-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 2px 12px 0 rgba(0,0,0,.1);
}

.image {
  width: 100%;
  height: 200px;
  object-fit: cover;
  display: block;
  cursor: pointer;
}

.image:hover {
  opacity: 0.8;
}

.image-info {
  padding: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.image-info span {
  flex: 1;
  min-width: 0;  /* 允许 flex 项目收缩到比内容更小 */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;  /* 防止操作按钮被压缩 */
}

.image-actions .el-button {
  padding: 2px;
}

/* 添加分页样式 */
.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.image-wrapper {
  position: relative;
}

.upload-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
}

.upload-progress {
  text-align: center;
  width: 80%;
}

.upload-icon {
  font-size: 32px;
  margin-bottom: 8px;
  animation: bounce 1s infinite;
}

.progress-bar {
  margin-top: 12px;
}

@keyframes bounce {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}
</style> 