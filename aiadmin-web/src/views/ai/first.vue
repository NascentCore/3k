<template>
  <div class="app-container">
    <el-form ref="form" :model="form" :rules="rules" label-width="22%">
      <el-form-item v-if="hfUrlshow" :label="$t('ai.trainingsource')" prop="hfUrl">
        <el-input v-model="form.hfUrl" style="width: 22%" :placeholder="$t('ai.inputtrainingsource')" />
        <span style="color: #000000;margin-left: 10px;">{{ $t("ai.mountpath") }}</span>
        <el-input v-model="form.datasetPath" style="width: 22%" :placeholder="$t('ai.inputmountpath')" />
      </el-form-item>
      <el-form-item :label="$t('ai.ckptpath')" prop="ckptPath">
        <el-input v-model="form.ckptPath" style="width: 22%" :placeholder="$t('ai.inputckptpath')" />
        <span style="color: #000000;margin-left: 10px;">{{ $t("ai.capacity") }}</span>
        <el-input v-model="form.ckptVol" style="width: 22%" :placeholder="$t('ai.capacitymes')" type="number" oninput="form.ckptVol = form.ckptVol.replace(/[^0-9]/g,'')" />
        <span style="color: #000000;margin-left: 6px;">{{ $t("ai.capacityvol") }}</span>
      </el-form-item>
      <el-form-item :label="$t('ai.ckptsavepath')" prop="modelPath">
        <el-input v-model="form.modelPath" style="width: 22%" :placeholder="$t('ai.inputckptsavepath')" />
        <span style="color: #000000;margin-left: 10px;">{{ $t("ai.capacity") }}</span>
        <el-input v-model="form.modelVol" style="width: 22%" :placeholder="$t('ai.capacitymes')" type="number" oninput="form.ckptVol = form.ckptVol.replace(/[^0-9]/g,'')" />
        <span style="color: #000000;margin-left: 6px;">{{ $t("ai.capacityvol") }}</span>
      </el-form-item>
      <el-form-item label="GPU：" prop="gpuNumber">
        <el-input v-model="form.gpuNumber" :disabled="true" style="width: 60px" placeholder="1" />
        <el-select v-model="form.gpuType" style="width: 23%" :placeholder="$t('ai.gpumodel')" @change="changeGpuType">
          <el-option v-for="item in gpus" :key="item.gpuProd" :label="item.gpuProd" :value="item.gpuProd" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('ai.containername')" prop="imagePath">
        <el-input v-model="form.imagePath" style="width: 40%" :placeholder="$t('ai.inputcontainername')" />
      </el-form-item>
      <el-form-item :label="$t('ai.jobtype')" prop="jobType">
        <el-select v-model="form.jobType" :placeholder="$t('ai.typeinfo')">
          <el-option label="MPI" value="MPI" />
        </el-select>
      </el-form-item>
      <el-form-item :label="$t('ai.stopcondition')" prop="stopType">
        <el-radio v-model="form.stopType" label="0" disabled @change="agreeChange">{{ $t("ai.stopradio1") }}</el-radio>
        <el-radio v-model="form.stopType" label="1" disabled @change="agreeChange">{{ $t("ai.stopradio2") }}</el-radio>
        <el-input v-if="timestatus" v-model="form.stopTime"  @input="onInputChange" type="number" style="width: 60px" placeholder="1" />
        <span v-if="timestatus" style="color: #000000;margin-left: 10px;">{{ $t("ai.hour") }}</span>
      </el-form-item>
      <el-form-item>
        <el-button v-loading.fullscreen.lock="fullscreenLoading" type="primary" style="width: 340px" :element-loading-text="$t('ai.submiting')" @click="doSubmit">{{ $t("ai.submit") }}</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script>
import { add, getAllGpuType } from '@/api/system/userJob'

export default {
  name: 'Userai',
  data() {
    const validValue = (rule, value, callback) => {
      if (value === '' || value === null) {
        callback(new Error(this.$t('login.emailmes')))
      } else {
        callback()
      }
    }
    return {
      height: document.documentElement.clientHeight - 180 + 'px;',
      gpus: [],
      form: {
        jobId: null,
        jobName: 'job1',
        gpuNumber: 1,
        gpuType: null,
        ckptPath: null,
        ckptVol: null,
        modelPath: null,
        modelVol: null,
        imagePath: null,
        hfUrl: null,
        datasetPath: null,
        jobType: null,
        stopType: '1',
        stopTime: 5,
        workStatus: null,
        obtainStatus: null
      },
      rules: {
        ckptPath: [
          { required: true, message: this.$t('ai.valuenotnull'), trigger: 'blur' }
        ],
        modelPath: [
          { required: true, message: this.$t('ai.valuenotnull'), trigger: 'blur' }
        ],
        gpuNumber: [
          { required: true, message: this.$t('ai.valuenotnull'), trigger: 'blur', validator: validValue }
        ],
        imagePath: [
          { required: true, message: this.$t('ai.valuenotnull'), trigger: 'blur' }
        ],
        jobType: [
          { required: true, message: this.$t('ai.valuenotnull'), trigger: 'blur' }
        ],
        stopType: [
          { required: true, message: this.$t('ai.valuenotnull'), trigger: 'blur' }
        ]
      },
      fullscreenLoading: false,
      timestatus: true,
      radio1: '1',
      hfUrlshow: false
    }
  },
  computed: {
  },
  created() {
    this.getGpuType()
  },
  mounted: function() {
    const that = this
    window.onresize = function temp() {
      that.height = document.documentElement.clientHeight - 180 + 'px;'
    }
  },
  methods: {
    onInputChange(value) {
      if (value > 5) {
        this.form.stopTime = 5
      }
    },
    doSubmit() {
      this.$refs['form'].validate((valid) => {
        if (valid) {
          this.fullscreenLoading = true
          add(this.form).then(res => {
            this.fullscreenLoading = false
            this.$router.replace('/ai/index')
          }).catch(err => {
            this.fullscreenLoading = false
            console.log(err.response.data.message)
          })
        }
      })
    },
    changeGpuType(value) {
      this.form.gpuType = value
    },
    // 获取gpu型号数据
    getGpuType() {
      getAllGpuType().then(res => {
        this.gpus = res
      }).catch(() => { })
    },
    agreeChange: function(val) {
      this.timestatus = (val === '1')
    }
  }
}
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
::v-deep input::-webkit-inner-spin-button {
  -webkit-appearance: none !important;
}
::v-deep input[type="number"] {
  -moz-appearance: textfield !important;
}

</style>
