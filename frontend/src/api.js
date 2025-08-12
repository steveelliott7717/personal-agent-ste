// frontend/src/api.js
const API_BASE = (
    import.meta.env.PROD
        ? '/app/api'
        : (import.meta.env.VITE_API_BASE_URL || '/app/api')
).replace(/\/+$/, '')

async function _json(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body ?? {})
    })
    if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status}: ${txt || res.statusText}`)
    }
    return res.json()
}

export async function sendRequest(query, extra = {}) {
    return _json(`${API_BASE}/request`, { query, ...extra })
}

export async function repoQuery(taskOrQuestion, extra = {}) {
    // task/question/q accepted; defaults k=8
    const body = { task: taskOrQuestion, k: 8, ...extra }
    return _json(`${API_BASE}/repo/query`, body)
}
