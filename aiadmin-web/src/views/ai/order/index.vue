<template>
  <div class="app-container">
    <!--工具栏-->
    <div class="head-container">
      <div v-if="crud.props.searchToggle">
        <!-- 搜索 -->
        <label class="el-form-item-label">{{ $t('order.ordernum') }}</label>
        <el-input v-model="query.outTradeNo" clearable :placeholder="$t('order.ordernum')" style="width: 185px;" class="filter-item" @keyup.enter.native="crud.toQuery" />
        <label class="el-form-item-label">{{ $t('order.jobname') }}</label>
        <el-input v-model="query.jobName" clearable :placeholder="$t('order.jobname')" style="width: 185px;" class="filter-item" @keyup.enter.native="crud.toQuery" />
        <rrOperation :crud="crud" />
      </div>
      <!--表格渲染-->
      <el-table ref="table" v-loading="crud.loading" :data="crud.data" size="small" style="width: 100%;" @selection-change="crud.selectionChangeHandler">
        <el-table-column prop="outTradeNo" :label="$t('order.ordernum')" />
        <el-table-column prop="subject" :label="$t('order.goodname')" />
        <el-table-column prop="totalAmount" :label="$t('order.payamount')" />
        <el-table-column prop="status" :label="$t('order.paystatus')">
          <template slot-scope="scope">
            <el-tag :type="scope.row.status === 0 ? 'danger' : 'success'">{{ scope.row.status === 0 ? $t('order.nopay') : $t('order.paid') }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="body" :label="$t('order.gooddetail')" />
        <el-table-column prop="createTime" :label="$t('order.createtime')" />
        <el-table-column prop="updateTime" :label="$t('order.updatetime')" />
      </el-table>
      <!--分页组件-->
      <pagination />
    </div>
  </div>
</template>

<script>
import CRUD, { presenter, header, form, crud } from '@crud/crud'
import rrOperation from '@crud/RR.operation'
import pagination from '@crud/Pagination'

const defaultForm = { orderId: null, userId: null, tradeNo: null, body: null, subject: null, totalAmount: null, status: null, createTime: null, updateTime: null, outTradeNo: null, jobName: null }
export default {
  name: 'Order',
  components: { pagination, rrOperation },
  mixins: [presenter(), header(), form(defaultForm), crud()],
  cruds() {
    return CRUD({ title: 'AliPayController', url: 'api/order', idField: 'orderId', sort: 'orderId,desc', crudMethod: {}})
  },
  data() {
    return {
      rules: {
      }
    }
  },
  methods: {
    [CRUD.HOOK.beforeRefresh]() {
      return true
    }
  }
}
</script>

<style scoped>

</style>
