<template>
  <div class="app-container">
    <el-dialog :visible.sync="centerDialogVisible">
      <el-image :src="Background" />
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
          <el-tag :type="scope.row.workStatus === 0 ? '' : scope.row.workStatus === 1 ? 'danger' : 'success'">{{ scope.row.workStatus === 0 ? $t('jobinfo.runstatus') : scope.row.workStatus === 1 ? $t('jobinfo.runstatusfail'):$t('jobinfo.runstatussuc') }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="$t('jobinfo.operate')" align="center" fixed="right">
        <template slot-scope="scope">
          <el-button style="margin-right: 3px;" type="text" size="medium" @click="executeinfo(scope.row)">{{ $t('jobinfo.info') }}</el-button>
          <el-button v-if="scope.row.workStatus === 2" style="margin-right: 3px;" type="text" size="medium" @click="executedownload(scope.row.jobName)">{{ $t('router.resultdownload') }}</el-button>
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
      <pre><el-link type="primary" :href="downloadInfo">{{ downloadInfo }}</el-link></pre>
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
      Background: Background,
      delLoading: false,
      centerDialogVisible: false,
      downloadInfo: '',
      dialogdown: false
    }
  },
  activated() {
    this.crud.toQuery()
  },
  methods: {
    executeinfo(id) {
      this.centerDialogVisible = true
      setTimeout(() => {
        this.centerDialogVisible = false
      }, 9000)
    },
    executedownload(jobname) {
      this.dialogdown = true
      const path = 'https:' + '//' + 'sxwl-ai.oss-cn-beijing.aliyuncs.com' + '/' + jobname
      this.downloadInfo = path
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
