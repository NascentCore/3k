<template>
  <div>
    <svg-icon icon-class="feedback" @click="click" />
  </div>
</template>

<script>
export default {
  name: 'Doc',
  methods: {
    click() {
      window.open('https://tricorder.feishu.cn/share/base/form/shrcnIfP4YYdip9m4JflYVEVy4d', '_blank')
    }
  }
}
</script>
