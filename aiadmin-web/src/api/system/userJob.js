import request from '@/utils/request'

export function add(data) {
  return request({
    url: 'api/userJob',
    method: 'post',
    data
  })
}

export function del(ids) {
  return request({
    url: 'api/userJob/',
    method: 'delete',
    data: ids
  })
}

export function edit(data) {
  return request({
    url: 'api/userJob',
    method: 'put',
    data
  })
}

export function getAllGpuType() {
  return request({
    url: 'api/userJob/getGpuType',
    method: 'get'
  })
}

export default { add, edit, del }
