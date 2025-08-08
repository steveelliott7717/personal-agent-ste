<template>
  <div>
    <h1>Dashboard</h1>
    <button @click="startVoice">🎤 Speak</button>
    <input v-model="text" placeholder="Type request">
    <button @click="sendText">Send</button>
    <p v-if="result">Result: {{ result }}</p>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      text: '',
      result: ''
    }
  },
  methods: {
    async sendText() {
      const res = await axios.post('/api/request', new URLSearchParams({ query: this.text }))
      this.result = res.data.result
    },
    async startVoice() {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const chunks = []
      recorder.ondataavailable = e => chunks.push(e.data)
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        const formData = new FormData()
        formData.append('file', blob, 'voice.webm')
        const res = await axios.post('/api/voice', formData)
        this.result = res.data.result
      }
      recorder.start()
      setTimeout(() => recorder.stop(), 3000)
    }
  }
}
</script>
