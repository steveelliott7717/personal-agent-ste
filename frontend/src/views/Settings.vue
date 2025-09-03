<template>
  <div>
    <h1>Settings</h1>
    <label>
      Mode:
      <select v-model="mode" @change="updateMode">
        <option value="demo">Demo</option>
        <option value="live">Live</option>
      </select>
    </label>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return { mode: 'demo' }
  },
  methods: {
    async updateMode() {
      await axios.post(
        '/api/request',
        new URLSearchParams({ query: `set mode ${this.mode}` })
      )
    },
  },
}
</script>
