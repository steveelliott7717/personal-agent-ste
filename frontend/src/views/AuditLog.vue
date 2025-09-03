<template>
  <div>
    <h1>Audit Log</h1>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Query</th>
          <th>Agent</th>
          <th>Result</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="entry in logs" :key="entry.id">
          <td>{{ entry.created_at }}</td>
          <td>{{ entry.query }}</td>
          <td>{{ entry.agent }}</td>
          <td>{{ entry.result }}</td>
          <td>{{ entry.source }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return { logs: [] }
  },
  async mounted() {
    const res = await axios.get(
      'https://YOUR-SUPABASE-REST-ENDPOINT/router_logs?select=*'
    )
    this.logs = res.data
  },
}
</script>
