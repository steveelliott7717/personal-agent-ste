<template>
  <div>
    <h1>Notifications</h1>
    <button @click="sendTest">Send Test Push</button>
  </div>
</template>

<script>
import axios from 'axios'
import { requestPermission, listenForMessages } from '../firebase'

export default {
  mounted() {
    requestPermission()
    listenForMessages()
  },
  methods: {
    async sendTest() {
      await axios.post(
        '/api/request',
        new URLSearchParams({ query: 'send test notification' })
      )
    },
  },
}
</script>
