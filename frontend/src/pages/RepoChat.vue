<template>
  <main class="min-h-screen max-w-3xl mx-auto p-6 space-y-4">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-semibold">GPT-5 RMS Chat</h1>
      <nav class="text-sm">
        <RouterLink class="underline" :to="{ name: 'home' }">← Back to Main</RouterLink>
      </nav>
    </header>

    <div class="border rounded-lg overflow-hidden">
      <div class="p-4 h-[60vh] overflow-y-auto space-y-4">
        <div v-for="(m, i) in messages" :key="i" class="space-y-1">
          <div class="text-xs text-gray-500">{{ m.role === 'user' ? 'You' : 'Repo Agent' }}</div>
          <div
            class="whitespace-pre-wrap break-words p-3 rounded"
            :class="m.role === 'user' ? 'bg-gray-100' : 'bg-white border'"
          >{{ m.content }}</div>

          <!-- Show top citations if present -->
          <div v-if="m.citations?.length" class="text-xs text-gray-500">
            Sources: {{ m.citations.join('  ·  ') }}
          </div>
        </div>
      </div>

      <div class="border-t p-3 flex gap-2">
        <input
          v-model="input"
          @keydown.enter.exact.prevent="send"
          class="flex-1 border rounded px-3 py-2"
          placeholder="Ask anything about your repo…"
        />
        <button
          @click="send"
          :disabled="loading || !input.trim()"
          class="px-4 py-2 rounded bg-black text-white disabled:opacity-50"
        >
          {{ loading ? 'Thinking…' : 'Ask' }}
        </button>
      </div>
    </div>

    <details class="text-sm">
      <summary class="cursor-pointer">Options</summary>
      <div class="mt-2 grid grid-cols-1 md:grid-cols-3 gap-3">
        <label class="flex items-center gap-2">
          <span class="text-gray-600 text-sm">Path prefix</span>
          <input v-model="prefix" class="border rounded px-2 py-1 flex-1" placeholder="e.g., backend/ or frontend/" />
        </label>
        <label class="flex items-center gap-2">
          <span class="text-gray-600 text-sm">Top-K</span>
          <input v-model.number="k" type="number" min="4" max="24" class="border rounded px-2 py-1 w-24" />
        </label>
      </div>
    </details>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import { repoQuery } from '../api'

const input = ref('')
const messages = ref([
  { role: 'assistant', content: 'Hi! Ask me anything about your repository. I answer using your RMS (code memory) and cite files.', citations: [] }
])
const loading = ref(false)
const prefix = ref('backend/')
const k = ref(8)

async function send() {
  const text = input.value.trim()
  if (!text) return
  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true
  try {
    const res = await repoQuery(text, { path_prefix: prefix.value || null, k: k.value || 8 })
    // Build a compact citation line list if present
    const cites = (res.hits || []).slice(0, 3).map((h, idx) => `[${idx+1}] ${h.path}:${h.start_line}–${h.end_line}@${(h.commit_sha||'').slice(0,7)}`)
    messages.value.push({ role: 'assistant', content: res.answer || '(no answer)', citations: cites })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: `Error: ${e?.message || e}` })
  } finally {
    loading.value = false
  }
}
</script>
