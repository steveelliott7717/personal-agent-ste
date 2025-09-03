// frontend/src/api.js

const BASE = (import.meta.env && import.meta.env.VITE_API_BASE) || ''

/** Build an absolute URL from a path or return the input if it's already absolute */
function toURL(path) {
  if (/^https?:\/\//i.test(path)) return path
  return `${BASE}`.replace(/\/+$/, '') + '/' + `${path}`.replace(/^\/+/, '')
}

async function handle(res) {
  if (!res.ok) {
    let text
    try {
      text = await res.text()
    } catch {
      text = ''
    }
    throw new Error(
      `HTTP ${res.status} ${res.statusText}${text ? ` – ${text}` : ''}`
    )
  }
  const ct = res.headers.get('content-type') || ''
  return ct.includes('application/json') ? res.json() : res.text()
}

export async function apiGet(path, options = {}) {
  const url = toURL(path)
  const res = await fetch(url, { method: 'GET', ...options })
  return handle(res)
}

export async function apiPost(path, body, options = {}) {
  const url = toURL(path)
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body ?? {}),
    ...options,
  })
  return handle(res)
}

export async function apiPut(path, body, options = {}) {
  const url = toURL(path)
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  }
  const res = await fetch(url, {
    method: 'PUT',
    headers,
    body: JSON.stringify(body ?? {}),
    ...options,
  })
  return handle(res)
}

export async function apiDelete(path, options = {}) {
  const url = toURL(path)
  const res = await fetch(url, { method: 'DELETE', ...options })
  return handle(res)
}

/**
 * Compat shim for older code that used `sendRequest`.
 * Flexible signature:
 *   sendRequest(path, { method="GET", body, headers={}, query })
 * `query` can be an object that will be added to the URL as ?key=value.
 */
export async function sendRequest(path, opts = {}) {
  const { method = 'GET', body, headers = {}, query } = opts

  let url = toURL(path)
  if (query && typeof query === 'object') {
    const qs = new URLSearchParams(query).toString()
    if (qs) url += (url.includes('?') ? '&' : '?') + qs
  }

  const init = {
    method,
    headers: { 'Content-Type': 'application/json', ...headers },
  }
  if (body !== undefined && method !== 'GET' && method !== 'HEAD') {
    init.body = typeof body === 'string' ? body : JSON.stringify(body)
  }

  const res = await fetch(url, init)
  return handle(res)
}
