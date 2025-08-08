export async function sendQuery(query, audioBlob = null) {
  const form = new FormData()
  form.append('query', query)
  if (audioBlob) {
    form.append('audio', audioBlob, 'note.webm')
  }

  const res = await fetch('/api/request', {
    method: 'POST',
    body: form
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || 'Request failed'}`)
  }

  return res.json()
}
