self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', () => self.clients.claim())
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {}
  event.waitUntil(
    self.registration.showNotification(data.title || 'Personal Agent', {
      body: data.body || '',
      icon: '/icon-192.png',
    })
  )
})
