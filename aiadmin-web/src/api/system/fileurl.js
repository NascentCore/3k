import request from '@/utils/request'
import qs from 'qs'
export function getUrls(params) {
  return request({
    url: 'api/info/model_urls/' + '?' + qs.stringify(params, { indices: false }),
    method: 'get'
  })
}
export function getPayStatus(params) {
  return request({
    url: 'api/order/order_status/' + '?' + qs.stringify(params, { indices: false }),
    method: 'get'
  })
}
export function getPayInfo(params) {
  return request({
    url: 'api/order/order_info/' + '?' + qs.stringify(params, { indices: false }),
    method: 'get'
  })
}
