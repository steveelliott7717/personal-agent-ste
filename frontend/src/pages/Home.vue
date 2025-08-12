<template>
  <main class="min-h-screen max-w-2xl mx-auto p-6 space-y-4">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-semibold">Personal Agent</h1>
      <nav class="text-sm">
        <RouterLink class="underline" :to="{ name: 'repo' }">Open GPT-5 RMS Chat →</RouterLink>
      </nav>
    </header>

    <section class="space-y-2">
      <label class="block text-sm text-gray-600">Query</label>
      <textarea
        v-model="query"
        rows="3"
        class="w-full border rounded p-2"
        placeholder="Type something…"
      />
      <div class="flex gap-2">
        <button
          @click="run"
          :disabled="loading || !query.trim()"
          class="px-4 py-2 rounded bg-black text-white disabled:opacity-50"
        >
          {{ loading ? 'Running…' : 'Send to /app/api/request' }}
        </button>
        <button @click="clearOut" class="px-3 py-2 rounded border">Clear</button>
      </div>
    </section>

    <section v-if="error" class="border border-red-300 bg-red-50 p-3 rounded text-sm">
      <strong>Error:</strong> {{ error }}
    </section>

    <section v-if="result" class="border p-3 rounded text-sm whitespace-pre-wrap break-words">
      <strong>Result</strong>
      <pre class="mt-2">{{ pretty(result) }}</pre>
    </section>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import { sendRequest } from '../api'

const query = ref('hello')
const loading = ref(false)
const error = ref('')
const result = ref(null)

function pretty(obj) { try { return JSON.stringify(obj, null, 2) } catch { return String(obj) } }

async function run() {
  error.value = ''; result.value = null; loading.value = true
  try { result.value = await sendRequest(query.value) }
  catch (e) { error.value = e?.message || String(e) }
  finally { loading.value = false }
}
function clearOut() { error.value = ''; result.value = null }
</script>
