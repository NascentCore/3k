import { useStorage } from '@vueuse/core'
import { TMessage } from '@/types'
import dayjs from 'dayjs'


const messages = useStorage<TMessage[]>('messages_' + window.location.pathname, [
  {
    username: "chatGPT",
    msg: "你好！请问有什么可以帮助您的？",
    time: dayjs().format('HH:mm'),
    type: 0,
  },
])



export default function useMessages() {
  const addMessage = (message: TMessage) => {
    messages.value.push({
      username: message.username,
      msg: message.msg,
      time: message.time || dayjs().format('HH:mm'),
      type: message.type,
    })
  }

  const clearMessages = () => {
    messages.value = []
  }

  const getLastMessages = (num: number = 10) => {
    return messages.value.slice(-num)
  }

  return {
    messages,
    addMessage,
    clearMessages,
    getLastMessages
  }
}
