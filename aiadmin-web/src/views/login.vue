<template>
  <div class="home" style="background: #FFFAFA">
    <div class="navbar">
      <div class="right-menu">
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
      </div>
    </div>
    <div v-if="isnotregister" class="login">
      <el-form ref="loginForm" :model="loginForm" :rules="loginRules" label-position="left" label-width="0px" class="login-form">
        <h3 class="title">
          {{ $t("login.title") }}
        </h3>
        <el-form-item prop="username">
          <el-input v-model="loginForm.username" type="text" auto-complete="off" :placeholder="$t('login.username')">
            <svg-icon slot="prefix" icon-class="user" class="el-input__icon input-icon" />
          </el-input>
        </el-form-item>
        <el-form-item prop="password">
          <el-input v-model="loginForm.password" type="password" auto-complete="off" :placeholder="$t('login.password')" show-password @keyup.enter.native="handleLogin">
            <svg-icon slot="prefix" icon-class="password" class="el-input__icon input-icon" />
          </el-input>
        </el-form-item>

        <el-checkbox v-model="loginForm.rememberMe" style="margin:0 0 25px 0;">
          {{ $t("login.rememberme") }}
        </el-checkbox>
        <el-form-item style="width:100%;">
          <el-button :loading="loading" size="medium" type="primary" style="width:47%;" @click.native.prevent="handleLogin">
            <span v-if="!loading">{{ $t("login.login") }}</span>
            <span v-else>{{ $t("login.logining") }}</span>
          </el-button>
          <el-button size="medium" type="primary" style="width:47%;" @click.native.prevent="goRegister">
            <span>{{ $t("login.registersub") }}</span>
          </el-button>
        </el-form-item>
      </el-form>
      <!--  底部  -->
      <div v-if="$store.state.settings.showFooter" id="el-login-footer">
        <span v-html="$store.state.settings.footerTxt" />
        <span v-if="$store.state.settings.caseNumber"> ⋅ </span>
        <a href="https://beian.miit.gov.cn/#/Integrated/index" target="_blank">{{ $store.state.settings.caseNumber }}</a>
      </div>
    </div>

    <div v-else class="login">
      <el-form ref="registerForm" :model="registerForm" :rules="registerrules" label-position="left" label-width="0px" class="login-form">
        <h3 class="title">
          {{ $t("login.registertitle") }}
        </h3>
        <el-form-item prop="email">
          <el-input v-model="registerForm.email" type="text" auto-complete="off" :placeholder="$t('login.registeremail')" />
        </el-form-item>
        <el-form-item prop="codemes">
          <el-input v-model="registerForm.codemes" type="text" auto-complete="off" :placeholder="$t('login.registercodemes')" style="width: 46%">
            <svg-icon slot="prefix" icon-class="message" class="el-input__icon input-icon" />
          </el-input>
          <el-button :loading="codeLoading" :disabled="isDisabled" size="medium" style="width: 52%" @click="sendCode"><span>{{ buttonName }}</span></el-button>
        </el-form-item>
        <el-form-item prop="password1">
          <el-input v-model="registerForm.password1" type="password" auto-complete="off" :placeholder="$t('login.registerpd')" show-password>
            <svg-icon slot="prefix" icon-class="password" class="el-input__icon input-icon" />
          </el-input>
        </el-form-item>
        <el-form-item prop="password2">
          <el-input v-model="registerForm.password2" type="password" auto-complete="off" :placeholder="$t('login.registerpd2')" show-password>
            <svg-icon slot="prefix" icon-class="password" class="el-input__icon input-icon" />
          </el-input>
        </el-form-item>

        <el-checkbox v-model="registerForm.readMe" style="margin:0 0 25px 0;">
          {{ $t("login.registerreadMe") }}
        </el-checkbox>
        <el-form-item style="width:100%;">
          <el-button :loading="registerloading" size="medium" type="primary" style="width:100%;" @click.native.prevent="handleregister">
            <span v-if="!registerloading"> {{ $t("login.registersub") }}</span>
            <span v-else>{{ $t("login.registersubing") }}</span>
          </el-button>
        </el-form-item>
        <el-form-item style="width:100%;">
          <el-button :loading="loading" size="medium" type="primary" plain style="width:100%;" @click.native.prevent="backlogin">
            <span>{{ $t("login.registerback") }}</span>
          </el-button>
        </el-form-item>
      </el-form>
      <!--  底部  -->
      <div v-if="$store.state.settings.showFooter" id="el-login-footer">
        <span v-html="$store.state.settings.footerTxt" />
        <span v-if="$store.state.settings.caseNumber"> ⋅ </span>
        <a href="https://beian.miit.gov.cn/#/Integrated/index" target="_blank">{{ $store.state.settings.caseNumber }}</a>
      </div>
    </div>
  </div>
</template>

<script>
import { encrypt } from '@/utils/rsaEncrypt'
import Config from '@/settings'
import { registerUser, sendEmail } from '@/api/login'
import Cookies from 'js-cookie'
import qs from 'qs'
import { validEmail } from '@/utils/validate'
export default {
  name: 'Login',
  data() {
    var validatePass2 = (rule, value, callback) => {
      if (value === '') {
        callback(new Error(this.$t('login.pdmes')))
      } else if (value !== this.registerForm.password1) {
        callback(new Error(this.$t('login.pdmesnosame')))
      } else {
        callback()
      }
    }
    const validMail = (rule, value, callback) => {
      if (value === '' || value === null) {
        callback(new Error(this.$t('login.emailmes')))
      } else if (validEmail(value)) {
        callback()
      } else {
        callback(new Error(this.$t('login.emailmeserr')))
      }
    }
    return {
      language: this.$i18n.locale,
      codeUrl: '',
      cookiePass: '',
      loginForm: {
        username: '',
        password: '',
        rememberMe: false,
        code: '',
        uuid: ''
      },
      registerForm: {
        password1: '',
        password2: '',
        readMe: false,
        email: '',
        codemes: ''
      },
      buttonName: this.$t('login.registersendcode'), isDisabled: false, time: 60,
      codeLoading: false,
      isnotregister: true,
      loginRules: {
        username: [{ required: true, trigger: 'blur', message: this.$t('login.usernameru') }],
        password: [{ required: true, trigger: 'blur', message: this.$t('login.passwordru') }]
      },
      registerrules: {
        email: [
          { required: true, validator: validMail, trigger: 'blur' }
        ],
        codemes: [
          { required: true, message: this.$t('login.codemes'), trigger: 'blur' }
        ],
        password1: [
          { required: true, message: this.$t('login.pdmesnull'), trigger: 'blur' }
        ],
        password2: [
          { required: true, validator: validatePass2, trigger: 'blur' }
        ]
      },
      loading: false,
      registerloading: false,
      redirect: undefined
    }
  },
  watch: {
    $route: {
      handler: function(route) {
        const data = route.query
        if (data && data.redirect) {
          this.redirect = data.redirect
          delete data.redirect
          if (JSON.stringify(data) !== '{}') {
            this.redirect = this.redirect + '&' + qs.stringify(data, { indices: false })
          }
        }
      },
      immediate: true
    }
  },
  created() {
    // 获取用户名密码等Cookie
    this.getCookie()
    // token 过期提示
    this.point()
  },
  methods: {
    backlogin() {
      this.isnotregister = true
      this.buttonName = this.$t('login.registersendcode')
      this.$refs.registerForm.hidden
      this.$refs['registerForm'].resetFields()
    },
    goRegister() {
      this.isnotregister = false
      this.$refs.loginForm.hidden
      this.$refs['loginForm'].resetFields()
      this.resetForm()
    },
    sendCode() {
      if (this.registerForm.email) {
        this.codeLoading = true
        this.buttonName = this.$t('login.registersendcode1')
        const _this = this
        sendEmail(this.registerForm.email).then(res => {
          this.$message({
            showClose: true,
            message: _this.$t('login.sendmesafter'),
            type: 'success'
          })
          this.codeLoading = false
          this.isDisabled = true
          this.buttonName = this.time-- + _this.$t('login.aftersec')
          this.timer = window.setInterval(function() {
            _this.buttonName = _this.time + _this.$t('login.aftersecre')
            --_this.time
            if (_this.time < 0) {
              _this.buttonName = _this.$t('login.registersendcode2')
              _this.time = 60
              _this.isDisabled = false
              window.clearInterval(_this.timer)
            }
          }, 1000)
        }).catch(err => {
          _this.resetForm()
          _this.codeLoading = false
          console.log(err.response.data.message)
        })
      } else {
        this.$message.error(this.$t('login.emailmes'))
      }
    },
    resetForm() {
      window.clearInterval(this.timer)
      this.time = 60
      this.buttonName = this.$t('login.registersendcode')
      this.isDisabled = false
      this.registerForm = { password1: '', password2: '', email: '', codemes: '' }
    },
    getCookie() {
      const username = Cookies.get('username')
      let password = Cookies.get('password')
      const rememberMe = Cookies.get('rememberMe')
      // 保存cookie里面的加密后的密码
      this.cookiePass = password === undefined ? '' : password
      password = password === undefined ? this.loginForm.password : password
      this.loginForm = {
        username: username === undefined ? this.loginForm.username : username,
        password: password,
        rememberMe: rememberMe === undefined ? false : Boolean(rememberMe),
        code: ''
      }
    },
    handleLogin() {
      this.$refs.loginForm.validate(valid => {
        const user = {
          username: this.loginForm.username,
          password: this.loginForm.password,
          rememberMe: this.loginForm.rememberMe
        }
        if (user.password !== this.cookiePass) {
          user.password = encrypt(user.password)
        }
        if (valid) {
          this.loading = true
          if (user.rememberMe) {
            Cookies.set('username', user.username, { expires: Config.passCookieExpires })
            Cookies.set('password', user.password, { expires: Config.passCookieExpires })
            Cookies.set('rememberMe', user.rememberMe, { expires: Config.passCookieExpires })
          } else {
            Cookies.remove('username')
            Cookies.remove('password')
            Cookies.remove('rememberMe')
          }
          this.$store.dispatch('Login', user).then(() => {
            this.loading = false
            this.$router.push({ path: this.redirect || '/' })
          }).catch(() => {
            this.loading = false
          })
        } else {
          console.log('error submit!!')
          return false
        }
      })
    },
    point() {
      const point = Cookies.get('point') !== undefined
      if (point) {
        this.$notify({
          title: this.$t('message.title'),
          message: this.$t('message.message'),
          type: 'warning',
          duration: 5000
        })
        Cookies.remove('point')
      }
    },

    handleSetLanguage(lang) {
      this.$i18n.locale = lang
      this.language = lang
      const icon = lang === 'zh' ? '/favicon.ico' : '/favicon-en.ico'
      this.changeFavIcon(icon)
      window.document.title = this.$t('router.title')
      this.buttonName = this.$t('login.registersendcode')
      this.$message({
        message: this.$t('message.switchlanguage').toString(),
        type: 'success',
        duration: 1000
      })
    },
    handleregister() {
      this.$refs['registerForm'].validate((valid) => {
        if (valid) {
          this.registerloading = true
          registerUser(this.registerForm).then(res => {
            const user = {
              username: this.registerForm.email,
              password: encrypt(this.registerForm.password1),
              rememberMe: false,
              code: '',
              uuid: ''
            }
            Cookies.remove('username')
            Cookies.remove('password')
            Cookies.remove('rememberMe')
            this.$store.dispatch('Login', user).then(() => {
              this.registerloading = false
              this.$router.push({ path: this.redirect || '/' })
            }).catch(() => {
              this.registerloading = false
            })
          }).catch(() => {
            this.registerloading = false
          })
        } else {
          return false
        }
      })
    },
    changeFavIcon(link) {
      let $favicon = document.querySelector('link[rel="icon"]')
      if ($favicon !== null) {
        $favicon.href = link
      } else {
        $favicon = document.createElement('link')
        $favicon.rel = 'icon'
        $favicon.href = link
        document.head.appendChild($favicon)
      }
    }
  }
}
</script>

<style rel="stylesheet/scss" lang="scss">
.home {
  height: 100%;
  background-size: cover;

  .login {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    background-size: cover;
  }

  .title {
    margin: 0 auto 30px auto;
    text-align: center;
    color: #707070;
  }

  .login-form {
    border-radius: 6px;
    background: #ffffff;
    width: 385px;
    padding: 25px 25px 5px 25px;

    .el-input {
      height: 38px;

      input {
        height: 38px;
      }
    }

    .input-icon {
      height: 39px;
      width: 14px;
      margin-left: 2px;
    }
  }

  .login-tip {
    font-size: 13px;
    text-align: center;
    color: #bfbfbf;
  }

  .login-code {
    width: 33%;
    display: inline-block;
    height: 38px;
    float: right;

    img {
      cursor: pointer;
      vertical-align: middle
    }
  }

  .navbar {
    height: 50px;
    overflow: hidden;
    position: relative;
    background-color: rgba(255, 255, 255, 0);

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
    }
  }
}
</style>
