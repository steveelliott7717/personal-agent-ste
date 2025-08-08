<template>
  <main class="wrap">
    <h1>Personal Agents</h1>

    <form class="bar" @submit.prevent="onSubmit">
      <input
        v-model="query"
        type="text"
        placeholder="Ask me anything… (e.g., 'list expenses', 'log meal eggs and toast')"
        autocomplete="off"
      />
      <button :disabled="loading">{{ loading ? 'Working…' : 'Send' }}</button>
    </form>

    <div class="row">
      <label class="ck">
        <input type="checkbox" v-model="demoMode" @change="persistDemoMode" />
        Demo mode
      </label>
    </div>

    <section v-if="error" class="error">{{ error }}</section>

    <section v-if="response" class="card">
      <h2>{{ response.agent }} • {{ response.intent }}</h2>
      <p class="message" v-if="response.message">{{ response.message }}</p>
      <details v-if="response.data">
        <summary>See raw data</summary>
        <pre>{{ pretty(response.data) }}</pre>
      </details>
      <details v-if="response.meta">
        <summary>See meta</summary>
        <pre>{{ pretty(response.meta) }}</pre>
      </details>
    </section>

    <footer class="hint">
      Try: <code>list expenses</code> • <code>what did I spend today</code> •
      <code>log meal chicken salad</code> • <code>show workouts</code> •
      <code>grooming today</code>
    </footer>
  </main>
</template>

<script>
import { sendQuery } from './api'

const DEMO_KEY = 'demoMode'

export default {
  name: 'App',
  data() {
    return {
      query: '',
      loading: false,
      error: '',
      response: null,
      demoMode: loadDemo()
    }
  },
  mounted() {
    // Persist demo mode across tabs via localStorage
    window.addEventListener('storage', (e) => {
      if (e.key === DEMO_KEY) {
        this.demoMode = loadDemo()
      }
    })
  },
  methods: {
    pretty(obj) {
      return JSON.stringify(obj, null, 2)
    },
    persistDemoMode() {
      try {
        localStorage.setItem(DEMO_KEY, this.demoMode ? '1' : '0')
      } catch {}
    },
    async onSubmit() {
      this.error = ''
      this.response = null
      const q = this.query?.trim()
      if (!q) return

      try {
        this.loading = true
        const json = await sendQuery(q)
        // Always prefer the human-friendly message; fallback to JSON if missing
        this.response = {
          agent: json.agent || 'router',
          intent: json.intent || 'unknown',
          message:
            json.message ||
            this.guessMessage(json) ||
            'Request completed. (No message provided.)',
          data: json.data ?? null,
          meta: json.meta ?? null
        }
      } catch (err) {
        this.error = err?.message || String(err)
      } finally {
        this.loading = false
      }
    },
    guessMessage(json) {
      // minimal generic fallback if backend didn’t attach `message`
      if (Array.isArray(json)) return `Got ${json.length} item(s).`
      if (json && typeof json === 'object') return 'Done.'
      return null
    }
  }
}

function loadDemo() {
  try {
    return localStorage.getItem(DEMO_KEY) === '1'
  } catch {
    return true
  }
}
</script>

<style>
:root {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  color-scheme: light dark;
}
body, html, #app { margin: 0; height: 100%; }
.wrap { max-width: 820px; margin: 0 auto; padding: 20px; }
.bar { display: flex; gap: 8px; margin: 12px 0; }
.bar input { flex: 1; padding: 10px 12px; font-size: 16px; }
.bar button { padding: 10px 14px; font-weight: 600; }
.card { border: 1px solid #4443; border-radius: 10px; padding: 14px; }
.message { font-size: 16px; margin: 8px 0 4px; }
.error { color: #c0392b; margin-top: 10px; }
.row { margin: 8px 0 14px; }
.ck { user-select: none; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; }
.hint { margin-top: 20px; color: #888; }
code { background: #0001; padding: 2px 6px; border-radius: 6px; }
</style>
