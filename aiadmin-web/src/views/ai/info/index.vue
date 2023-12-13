<template>
  <div class="app-container">
    <el-dialog :visible.sync="centerDialogVisible">
      <el-image :src="Background" />
    </el-dialog>
    <el-dialog :visible.sync="centerDialogPay" :title="$t('jobinfo.payinfo')" center :close-on-click-modal="false" :before-close="cancel">
      <el-form ref="form" :model="form" style="margin-top: 6px;margin-left: 30%" size="small" label-width="110px">
        <el-form-item :label="$t('jobinfo.goodname')" prop="subject">
          <span style="color: #C0C0C0;margin-left: 6px;">{{ subject }}</span>
        </el-form-item>
        <el-form-item :label="$t('jobinfo.paytotal')" prop="totalAmount">
          <span style="color: #C0C0C0;margin-left: 6px;">{{ amount }}元</span>
        </el-form-item>
        <el-form-item :label="$t('jobinfo.usedetail')" prop="body">
          <span style="color: #C0C0C0;margin-left: 6px;">{{ body }}</span>
        </el-form-item>
      </el-form>
      <!-- 定义一个展示二维码的div -->
      <div style="display: flex; justify-content: center">
        <!-- 二维码对象可以通过 ref 绑定 -->
        <div id="qrCode" ref="qrCodeDiv" />
      </div>
    </el-dialog>
    <!--表格渲染-->
    <el-table ref="table" v-loading="crud.loading" :data="crud.data" style="width: 100%;" @selection-change="crud.selectionChangeHandler">
      <el-table-column prop="jobName" :label="$t('jobinfo.jobName')" />
      <el-table-column prop="gpuNumber" :label="$t('jobinfo.gpunumber')" />
      <el-table-column prop="gpuType" :label="$t('jobinfo.gputype')" />
      <el-table-column prop="ckptPath" :label="$t('jobinfo.ckptpath')" />
      <el-table-column prop="modelPath" :label="$t('jobinfo.modelpath')" />
      <el-table-column prop="imagePath" :label="$t('jobinfo.beanName')" />
      <el-table-column prop="hfUrl" :label="$t('jobinfo.trainingsource')" />
      <el-table-column prop="datasetPath" :label="$t('jobinfo.mountpath')" />
      <el-table-column prop="jobType" :label="$t('jobinfo.jobtype')" />
      <el-table-column prop="workStatus" :label="$t('jobinfo.isPause')">
        <template slot-scope="scope">
          <el-tag :type="scope.row.workStatus === 1 ? 'danger' : scope.row.workStatus === 3 ? 'success' : ''">{{ scope.row.workStatus === 1 ? $t('jobinfo.runstatusfail') : scope.row.workStatus === 3 ? $t('jobinfo.runstatussuc'):$t('jobinfo.runstatus') }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="createTime" :label="$t('jobinfo.createtime')" />
      <el-table-column :label="$t('jobinfo.operate')" align="center" fixed="right">
        <template slot-scope="scope">
          <el-button style="margin-right: 3px;" type="text" size="medium" @click="executeinfo(scope.row)">{{ $t('jobinfo.info') }}</el-button>
          <el-button v-if="scope.row.workStatus === 3" v-loading.fullscreen.lock="fullscreenLoading" :element-loading-text="$t('jobinfo.modelload')" style="margin-right: 3px;" type="text" size="medium" @click="executedownload(scope.row.jobName)">{{ $t('router.resultdownload') }}</el-button>
          <el-popover :ref="scope.row.jobId" placement="top" width="200">
            <p>{{ $t('jobinfo.suredel') }}</p>
            <div style="text-align: right; margin: 0">
              <el-button size="mini" type="text" @click="$refs[scope.row.jobId].doClose()">{{ $t('jobinfo.cancel') }}</el-button>
              <el-button :loading="delLoading" type="primary" size="mini" @click="delMethod(scope.row.jobId)">{{ $t('jobinfo.sure') }}</el-button>
            </div>
            <el-button slot="reference" type="text" size="medium">{{ $t('jobinfo.delete') }}</el-button>
          </el-popover>
        </template>
      </el-table-column>
    </el-table>
    <el-dialog :visible.sync="dialogdown" :title="$t('jobinfo.downloadpath')" width="85%">
      <div v-for="(item, index) in modelurls" :key="index">
        <el-link type="primary" :href="item.fileUrl" style="margin-top: 10px">{{ item.fileUrl }}</el-link>
      </div>
    </el-dialog>
    <!--分页组件-->
    <pagination />
  </div>
</template>

<script>
import crudJob from '@/api/system/userJob'
import CRUD, { presenter, header, form, crud } from '@crud/crud'
import pagination from '@crud/Pagination.vue'
import Background from '@/assets/images/GPUpic.png'
import { getUrls, getPayStatus, getPayInfo } from '@/api/system/fileurl'
import QRCode from 'qrcodejs2'

const defaultForm = { jobId: null, jobName: null, gpuNumber: null, gpuType: null, ckptPath: null, ckptVol: null, modelPath: null, modelVol: null, imagePath: null, hfUrl: null, datasetPath: null, jobType: null, stopType: null, stopTime: null, workStatus: null, obtainStatus: null, createTime: null, updateTime: null }
export default {
  name: 'Infoai',
  components: { pagination },
  cruds() {
    return CRUD({ title: '任务', url: 'api/userJob', crudMethod: { ...crudJob }})
  },
  mixins: [presenter(), header(), form(defaultForm), crud()],
  data() {
    return {
      monitor: null,
      fullscreenLoading: false,
      Background: Background,
      delLoading: false,
      centerDialogVisible: false,
      downloadInfo: '',
      dialogdown: false,
      modelurls: [],
      centerDialogPay: false,
      subject: '',
      amount: '',
      body: ''
    }
  },
  activated() {
    this.crud.toQuery()
  },
  methods: {
    cancel() {
      clearInterval(this.monitor)
      this.centerDialogPay = false
    },
    createQRCode(id) {
      this.$nextTick(async() => {
        await this.$refs.qrCodeDiv
        this.$refs.qrCodeDiv.innerHTML = ''
        new QRCode(this.$refs.qrCodeDiv, {
          text: id,
          width: 200,
          height: 200,
          colorDark: '#333333',
          colorLight: '#ffffff',
          correctLevel: QRCode.CorrectLevel.H
        })
      })
    },
    executeinfo(id) {
      this.centerDialogVisible = true
      setTimeout(() => {
        this.centerDialogVisible = false
      }, 9000)
    },
    executedownload(jobname) {
      this.fullscreenLoading = true
      getPayInfo({ jobName: jobname }).then(res => {
        if (res.state === '1') {
          this.fullscreenLoading = false
          this.getUrls(jobname)
        } else {
          this.createQRCode(res.qrCode)
          this.subject = res.subject
          this.amount = res.totalAmount
          this.body = res.body
          this.fullscreenLoading = false
          this.centerDialogPay = true
          this.monitor = window.setInterval(() => {
            getPayStatus({ jobName: jobname }).then(res => {
              if (res.state === '1') {
                this.cancel()
                this.getUrls(jobname)
              }
            })
          }, 2000)
        }
      }).catch(() => { })
    },
    getUrls(name) {
      getUrls({ jobName: name }).then(res => {
        this.modelurls = res
        this.dialogdown = true
      }).catch(() => { })
    },
    delMethod(id) {
      this.delLoading = true
      crudJob.del([id]).then(() => {
        this.delLoading = false
        this.$refs[id].doClose()
        this.crud.dleChangePage(1)
        this.crud.toQuery()
      }).catch(() => {
        this.delLoading = false
        this.$refs[id].doClose()
      })
    }
  }
}
</script>
<style rel="stylesheet/scss" lang="scss">
</style>
