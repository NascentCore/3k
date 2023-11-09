<template>
  <div class="app-container">
    <!--工具栏-->
    <div class="head-container">
      <!--表格渲染-->
      <el-table ref="table" v-loading="crud.loading" :data="crud.data" style="width: 100%;" @selection-change="crud.selectionChangeHandler">
        <el-table-column prop="accessKey" :label="$t('componyuser.accesskey')" />
        <el-table-column prop="companyName" :label="$t('componyuser.name')" />
        <el-table-column prop="companyPhone" :label="$t('componyuser.componyphone')" />
        <el-table-column prop="email" :label="$t('componyuser.componyemail')" />
        <el-table-column prop="createTime" :label="$t('componyuser.createtime')" />
      </el-table>
      <!--分页组件-->
      <pagination />
    </div>
  </div>
</template>

<script>
import crudUser from '@/api/system/user'
import CRUD, { presenter, header, form, crud } from '@crud/crud'
import pagination from '@crud/Pagination'

const defaultForm = { userId: null, deptId: null, username: null, nickName: null, gender: null, phone: null, email: null, avatarName: null, avatarPath: null, password: null, isAdmin: null, enabled: null, createBy: null, updateBy: null, pwdResetTime: null, createTime: null, updateTime: null, userType: null, accessKey: null }
export default {
  name: 'User',
  components: { pagination },
  mixins: [presenter(), header(), form(defaultForm), crud()],
  cruds() {
    return CRUD({ title: 'usercontroller', url: 'api/users', idField: 'userId', sort: 'userId,desc', crudMethod: { ...crudUser }})
  },
  data() {
    return {}
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
