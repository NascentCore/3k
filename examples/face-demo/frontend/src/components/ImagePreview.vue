<template>
  <el-dialog
    v-model="visible"
    :show-close="true"
    width="80%"
    class="image-preview-dialog"
    destroy-on-close
    @close="handleClose"
  >
    <div class="preview-container">
      <div class="image-container">
        <div class="image-wrapper">
          <img :src="imageUrl" class="preview-image" @load="onImageLoad" ref="previewImageRef"/>
          <div 
            v-for="(face, index) in faces" 
            :key="index"
            class="face-box"
            :style="getFaceBoxStyle(face)"
          >
          </div>
        </div>
      </div>
      <div class="faces-container">
        <h3>人脸标签</h3>
        <div class="faces-list">
          <div 
            v-for="(face, index) in faces" 
            :key="index"
            class="face-item"
          >
            <div 
            class="face-crop">
              <img 
                :src="imageUrl"
                :style="getFaceCropImageStyle(face)"
                class="face-crop-image"
              />
            </div>
            <div class="face-tag">
              <span class="tag-text">{{ face.tag }}</span>
              <el-button
                type="primary"
                link
                class="edit-btn"
                @click="handleEditTag(index, face)"
              >
                <el-icon><Edit /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </el-dialog>
</template>

<script>
import { ref, defineProps, defineEmits, watch, onMounted, onUnmounted } from 'vue'
import { Edit } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

export default {
  name: 'ImagePreview',
  components: {
    Edit
  },
  props: {
    modelValue: {
      type: Boolean,
      default: false
    },
    imageUrl: {
      type: String,
      required: true
    },
    imageId: {
      type: String,
      required: true
    },
    faces: {
      type: Array,
      default: () => [],
    }
  },
  emits: ['update:modelValue', 'close', 'update-face-tag', 'refresh'],
  setup(props, { emit }) {
    const visible = ref(props.modelValue)
    const previewImageRef = ref(null)
    const imageSize = ref({ width: 0, height: 0 })

    const handleKeydown = (e) => {
      if (e.key === 'Escape' && visible.value) {
        handleClose()
      }
    }

    const handleClose = () => {
      visible.value = false
      emit('update:modelValue', false)
      emit('close')
    }

    watch(() => props.modelValue, (val) => {
      visible.value = val
    })

    onMounted(() => {
      document.addEventListener('keydown', handleKeydown)
    })

    onUnmounted(() => {
      document.removeEventListener('keydown', handleKeydown)
    })

    const onImageLoad = () => {
      if (previewImageRef.value) {
        imageSize.value = {
          width: previewImageRef.value.width,
          height: previewImageRef.value.height
        }
      }
    }

    const getFaceBoxStyle = (face) => {
      const { width, height } = imageSize.value
      if (!width || !height) return {}

      // 获取图片的原始尺寸和显示尺寸
      const imgElement = previewImageRef.value
      if (!imgElement) return {}

      // 计算缩放比例
      const scaleX = imgElement.clientWidth / imgElement.naturalWidth
      const scaleY = imgElement.clientHeight / imgElement.naturalHeight

      // 应缩放比例到坐标
      const x = face.face_coords.x1 * scaleX
      const y = face.face_coords.y1 * scaleY
      const w = (face.face_coords.x2 - face.face_coords.x1) * scaleX
      const h = (face.face_coords.y2 - face.face_coords.y1) * scaleY

      return {
        left: `${x}px`,
        top: `${y}px`,
        width: `${w}px`,
        height: `${h}px`
      }
    }
    const getFaceCropImageStyle = (face) => {
      const imgElement = previewImageRef.value;
      if (!imgElement) return {};

      const { x1, y1, x2, y2 } = face.face_coords;
      const faceWidth = x2 - x1;
      const faceHeight = y2 - y1;

      // 裁剪容器的尺寸（正方形）
      const containerSize = 100;

      // 确保根据人脸的长宽比计算缩放比例
      const scale =
        faceWidth > faceHeight
          ? containerSize / faceWidth // 人脸宽大于高，按宽缩放
          : containerSize / faceHeight; // 人脸高大于宽，按高缩放

      // 计算缩放后的偏移量，保证人脸居中
      const offsetX = -x1 * scale + (containerSize - faceWidth * scale) / 2;
      const offsetY = -y1 * scale + (containerSize - faceHeight * scale) / 2;

      return {
        position: 'absolute',
        width: `${imgElement.naturalWidth * scale}px`,
        height: `${imgElement.naturalHeight * scale}px`,
        transform: `translate(${offsetX}px, ${offsetY}px)`,
        transformOrigin: '0 0',
      };
    }

    const handleEditTag = async (index, face) => {
      try {
        if (!props.imageId) {
          throw new Error('图片ID未定义，请确保正确传入 image-id 属性')
        }

        const { value: tag } = await ElMessageBox.prompt('请输入人脸标签', '编辑标签', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          inputValue: face.tag || '',
          inputPlaceholder: '请输入标签名称'
        })
        
        if (tag !== null) {
          const requestData = {
            image_id: props.imageId,
            face_index: index,
            tag: tag
          }
          console.log('Sending request with data:', requestData)
          
          const response = await fetch('/api/face/tag', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
          })

          if (!response.ok) {
            throw new Error('标签更新失败')
          }

          emit('update-face-tag', { index, tag })
          emit('refresh')
          ElMessage.success('标签更新成功')
        }
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error(error.message || '标签更新失败')
        }
      }
    }

    return {
      visible,
      previewImageRef,
      onImageLoad,
      getFaceBoxStyle,
      handleClose,
      getFaceCropImageStyle,
      handleEditTag
    }
  }
}
</script>

<style scoped>
.image-preview-dialog :deep(.el-dialog__body) {
  padding: 0;
}

.preview-container {
  display: flex;
  min-height: 500px;
}

.image-container {
  flex: 2;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.image-wrapper {
  position: relative;
  display: inline-block;
}

.preview-image {
  max-width: 100%;
  max-height: 70vh;
  object-fit: contain;
}

.face-box {
  position: absolute;
  border: 2px solid #67C23A;
  box-sizing: border-box;
  pointer-events: none;
}

.faces-container {
  flex: 1;
  padding: 16px;
  border-left: 1px solid #eee;
  background: #fff;
  overflow-y: auto;
  min-width: 360px;
}

.faces-list {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 20px;
}

.face-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 4px;
  border: 1px solid #eee;
  border-radius: 4px;
  background: #fff;
}

.face-crop {
  width: 100%;
  aspect-ratio: 1;
  overflow: hidden;
  position: relative;
  background: #f5f5f5;
  border-radius: 4px;
  align-items: center;
  justify-content: center;
}

.face-crop-image {
  position: absolute;
}

.face-index {
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.faces-container h3 {
  margin: 0;
  padding-bottom: 12px;
  border-bottom: 1px solid #eee;
  color: #333;
  font-size: 16px;
}

.face-tag {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 4px 8px;
}

.tag-text {
  font-size: 12px;
  flex: 1;
  margin-right: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tag-text:not(:empty) {
  color: #303133;
  font-weight: 500;
}

.tag-text:empty::before {
  content: '未标记';
  color: #909399;
  font-style: italic;
}

.edit-btn {
  padding: 2px;
  font-size: 14px;
}
</style> 