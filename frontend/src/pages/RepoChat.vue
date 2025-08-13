<template>
  <main class="min-h-screen max-w-3xl mx-auto p-6 space-y-4">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-semibold">GPT-5 RMS Chat</h1>
      <nav class="text-sm">
        <RouterLink class="underline" :to="{ name: 'home' }">← Back</RouterLink>
      </nav>
    </header>

    <!-- Route-level error boundary UI -->
    <div v-if="fatal" class="p-4 border bg-red-50 rounded text-sm">
      <strong>Repo Chat failed to load.</strong>
      <div class="mt-1">{{ String(fatal) }}</div>
      <a class="underline mt-2 inline-block" href="/app/">Go back</a>
    </div>

    <div v-else class="border rounded-lg overflow-hidden">
      <!-- RMS health banner -->
      <div v-if="healthError" class="p-3 bg-yellow-50 border-b text-sm">
        ⚠️ Repo chat API seems unavailable: {{ healthError }}
      </div>
      <div v-else-if="health" class="p-2 border-b text-xs text-gray-500">
        Model: {{ health.model }} · Endpoint OK
      </div>

      <!-- Messages -->
      <div class="p-4 h-[60vh] overflow-y-auto space-y-4">
        <div v-for="(m, i) in messages" :key="i" class="space-y-1">
          <div class="text-xs text-gray-500">{{ m.role === 'user' ? 'You' : 'Repo Agent' }}</div>
          <div
            class="whitespace-pre-wrap break-words p-3 rounded"
            :class="m.role === 'user' ? 'bg-gray-100' : 'bg-white border'"
          >{{ m.content }}</div>
          <div v-if="m.citations?.length" class="text-xs text-gray-500">
            Sources: {{ m.citations.join('  ·  ') }}
          </div>
        </div>
      </div>

      <!-- Input -->
      <div class="border-t p-3 flex flex-col gap-2 md:flex-row">
        <div class="flex-1 flex gap-2">
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
        <div class="flex gap-2 text-sm">
          <input v-model="prefix" class="border rounded px-2 py-1 w-44" placeholder="path prefix (e.g. backend/)" />
          <input v-model.number="k" type="number" min="4" max="24" class="border rounded px-2 py-1 w-20" />
        </div>
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref, onMounted, onErrorCaptured } from 'vue'

const fatal = ref(null)
onErrorCaptured((err) => { fatal.value = err; return false })

const input = ref('')
const messages = ref([
  { role: 'assistant', content: 'Hi! Ask me anything about your repository. I’ll answer using your RMS and cite files.', citations: [] }
])
const loading = ref(false)
const prefix = ref('backend/')
const k = ref(8)

const health = ref(null)
const healthError = ref('')

async function checkHealth() {
  try {
    const r = await fetch('/app/api/repo/health')
    if (!r.ok) throw new Error(await r.text())
    health.value = await r.json()
  } catch (e) {
    healthError.value = e?.message || String(e)
  }
}

onMounted(checkHealth)

async function send() {
  const text = input.value.trim()
  if (!text) return
  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true
  try {
    const body = { task: text, path_prefix: prefix.value || null, k: k.value || 8 }
    const res = await fetch('/app/api/repo/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) throw new Error(await res.text())
    const data = await res.json()
    const cites = (data.hits || []).slice(0, 3).map((h, idx) =>
      `[${idx+1}] ${h.path}:${h.start_line}–${h.end_line}@${(h.commit_sha||'').slice(0,7)}`
    )
    messages.value.push({ role: 'assistant', content: data.answer || '(no answer)', citations: cites })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: `Error: ${e?.message || String(e)}` })
  } finally {
    loading.value = false
  }
}
</script>
