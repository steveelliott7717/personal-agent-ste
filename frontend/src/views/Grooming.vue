<template>
  <div>
    <h1>Grooming</h1>
    <input v-model="task" placeholder="Log a grooming task" />
    <button @click="logTask">Log Task</button>
    <ul>
      <li v-for="g in logs" :key="g.id">{{ g.description }}</li>
    </ul>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return { task: '', logs: [] }
  },
  mounted() {
    this.loadLogs()
  },
  methods: {
    async logTask() {
      await axios.post(
        '/api/request',
        new URLSearchParams({ query: `log grooming ${this.task}` })
      )
      this.loadLogs()
    },
    async loadLogs() {
      const res = await axios.post(
        '/api/request',
        new URLSearchParams({ query: 'show grooming' })
      )
      this.logs = res.data.result
    },
  },
}
</script>
