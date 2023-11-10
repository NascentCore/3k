import request from '@/utils/request'
import qs from 'qs'
export function getUrls(params) {
  return request({
    url: 'api/info/model_urls/' + '?' + qs.stringify(params, { indices: false }),
    method: 'get'
  })
}
