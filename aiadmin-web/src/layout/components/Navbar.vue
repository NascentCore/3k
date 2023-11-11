<template>
  <div class="navbar">
    <hamburger id="hamburger-container" :is-active="sidebar.opened" class="hamburger-container" @toggleClick="toggleSideBar" />

    <breadcrumb id="breadcrumb-container" class="breadcrumb-container" />

    <div class="right-menu">
      <template v-if="device!=='mobile'" />
      <el-tooltip :content="$t('message.feedback')" effect="dark" placement="bottom">
        <Doc class="right-menu-item hover-effect" />
      </el-tooltip>
      <el-dropdown trigger="click" class=" languageswitch " @command="handleSetLanguage">
        <div>
          <span>{{ $t('message.changelanguage') }}</span>
        </div>
        <el-dropdown-menu slot="dropdown">
          <el-dropdown-item :disabled="language === 'zh'" command="zh">
            {{ $t('message.chinese') }}
          </el-dropdown-item>
          <el-dropdown-item :disabled="language === 'en'" command="en">
            {{ $t('message.english') }}
          </el-dropdown-item>
        </el-dropdown-menu>
      </el-dropdown>
      <el-dropdown class="avatar-container right-menu-item " trigger="click">
        <div class="avatar-wrapper">
          <img :src="user.avatarName ? baseApi + '/avatar/' + user.avatarName : Avatar" class="user-avatar">
          <i class="el-icon-caret-bottom" />
        </div>
        <el-dropdown-menu slot="dropdown">

          <span style="display:block;" @click="open">
            <el-dropdown-item divided>
              {{ $t('message.logout') }}
            </el-dropdown-item>
          </span>
        </el-dropdown-menu>
      </el-dropdown>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import Breadcrumb from '@/components/Breadcrumb'
import Hamburger from '@/components/Hamburger'
import Avatar from '@/assets/images/avatar.png'
import Doc from '@/components/Doc'

export default {
  components: {
    Breadcrumb,
    Hamburger,
    Doc
  },
  data() {
    return {
      Avatar: Avatar,
      dialogVisible: false,
      language: this.$i18n.locale
    }
  },
  computed: {
    ...mapGetters([
      'sidebar',
      'device',
      'user',
      'baseApi'
    ]),
    show: {
      get() {
        return this.$store.state.settings.showSettings
      },
      set(val) {
        this.$store.dispatch('settings/changeSetting', {
          key: 'showSettings',
          value: val
        })
      }
    }
  },
  methods: {
    mounted() {
      this.$i18n.locale = 'zh'
    },
    changeFavicon(link) {
      let $favicon = document.querySelector('link[rel="icon"]')
      // If a <link rel="icon"> element already exists,
      // change its href to the given link.
      if ($favicon !== null) {
        $favicon.href = link
        // Otherwise, create a new element and append it to <head>.
      } else {
        $favicon = document.createElement('link')
        $favicon.rel = 'icon'
        $favicon.href = link
        document.head.appendChild($favicon)
      }
    },
    handleSetLanguage(lang) {
      this.$i18n.locale = lang
      this.language = lang
      const icon = lang === 'zh' ? '/favicon.ico' : '/favicon-en.ico'
      this.changeFavicon(icon)
      window.document.title = this.$t('router.title')
      this.$message({
        message: this.$t('message.switchlanguage').toString(),
        type: 'success',
        duration: 1000
      })
    },
    toggleSideBar() {
      this.$store.dispatch('app/toggleSideBar')
    },
    open() {
      const that = this
      this.$confirm(that.$t('message.logoutinfo'), that.$t('message.logouttag'), {
        confirmButtonText: that.$t('message.logoutsure'),
        cancelButtonText: that.$t('message.logoutcancel'),
        type: 'warning'
      }).then(() => {
        this.logout()
      })
    },
    logout() {
      this.$store.dispatch('LogOut').then(() => {
        location.reload()
      })
    }
  }
}
</script>

<style lang="scss" scoped>
.international{
  background: #fff;
}
.navbar {
  height: 50px;
  overflow: hidden;
  position: relative;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0,21,41,.08);
  .hamburger-container {
    line-height: 46px;
    height: 100%;
    float: left;
    cursor: pointer;
    transition: background .3s;
    -webkit-tap-highlight-color:transparent;

    &:hover {
      background: rgba(0, 0, 0, .025)
    }
  }

  .breadcrumb-container {
    float: left;
  }

  .errLog-container {
    display: inline-block;
    vertical-align: top;
  }

  .right-menu {
    float: right;
    height: 100%;
    line-height: 50px;
    &:focus {
      outline: none;
    }
    .languageswitch {
      display: inline-block;
      padding: 0 8px;
      height: 100%;
      font-size: 14px;
      color: #00a0e9;
      vertical-align: text-bottom;
    }
    .right-menu-item {
      display: inline-block;
      padding: 0 8px;
      height: 100%;
      font-size: 18px;
      color: #5a5e66;
      vertical-align: text-bottom;

      &.hover-effect {
        cursor: pointer;
        transition: background .3s;

        &:hover {
          background: rgba(0, 0, 0, .025)
        }
      }
    }

    .avatar-container {
      margin-right: 30px;

      .avatar-wrapper {
        margin-top: 5px;
        position: relative;

        .user-avatar {
          cursor: pointer;
          width: 40px;
          height: 40px;
          border-radius: 10px;
        }

        .el-icon-caret-bottom {
          cursor: pointer;
          position: absolute;
          right: -20px;
          top: 25px;
          font-size: 12px;
        }
      }
    }
  }
}
</style>
