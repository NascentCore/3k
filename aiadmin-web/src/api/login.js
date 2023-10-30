import request from '@/utils/request'
import { encrypt } from '@/utils/rsaEncrypt'

export function sendEmail(data) {
  return request({
    url: 'api/code/sendEmail?email=' + data,
    method: 'post'
  })
}
export function login(username, password, code, uuid) {
  return request({
    url: 'auth/login',
    method: 'post',
    data: {
      username,
      password,
      code,
      uuid
    }
  })
}
export function registerUser(form) {
  const data = {
    password: encrypt(form.password1),
    email: form.email,
    username: form.email,
    enabled: 1
  }
  return request({
    url: 'api/users/registerUser/' + form.codemes,
    method: 'post',
    data
  })
}

export function getInfo() {
  return request({
    url: 'auth/info',
    method: 'get'
  })
}

export function getCodeImg() {
  return request({
    url: 'auth/code',
    method: 'get'
  })
}

export function logout() {
  return request({
    url: 'auth/logout',
    method: 'delete'
  })
}
