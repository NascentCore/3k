import request from '@/utils/request'
export function buildMenus() {
  return request({
    url: 'api/menus/build',
    method: 'get'
  })
}
