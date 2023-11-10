<template>
  <div class="app-container">
    <div class="head-container">
      <!--表格渲染-->
      <el-table ref="table" v-loading="crud.loading" :data="crud.data" size="small" style="width: 100%;" @selection-change="crud.selectionChangeHandler">
        <el-table-column prop="cpodId" :label="$t('hashrate.cpodid')" />
        <el-table-column prop="cpodVersion" :label="$t('hashrate.cpodv')" />
        <el-table-column prop="gpuVendor" :label="$t('hashrate.gpuvendor')" />
        <el-table-column prop="gpuProd" :label="$t('hashrate.gpuProd')" />
        <el-table-column prop="gpuTotal" :label="$t('hashrate.gputotal')" />
        <el-table-column prop="gpuAllocatable" :label="$t('hashrate.gpuAllocatable')" />
        <el-table-column prop="createTime" :label="$t('hashrate.createtime')" />
        <el-table-column prop="updateTime" :label="$t('hashrate.updateTime')" />
      </el-table>
      <!--分页组件-->
      <pagination />
    </div>
  </div>
</template>

<script>
import CRUD, { presenter, header, form, crud } from '@crud/crud'
import pagination from '@crud/Pagination'

const defaultForm = { mainId: null, cpodId: null, cpodVersion: null, gpuVendor: null, gpuProd: null, gpuTotal: null, gpuAllocatable: null, createTime: null, updateTime: null, userId: null }
export default {
  name: 'CpodMain',
  components: { pagination },
  mixins: [presenter(), header(), form(defaultForm), crud()],
  cruds() {
    return CRUD({ title: 'cpod信息', url: 'api/cpod', idField: 'mainId', sort: 'mainId,desc', crudMethod: { }})
  },
  data() {
    return {
    }
  },
  methods: {
    // 钩子：在获取表格数据之前执行，false 则代表不获取数据
    [CRUD.HOOK.beforeRefresh]() {
      return true
    }
  }
}
</script>

<style scoped>

</style>
