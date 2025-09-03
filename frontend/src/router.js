// frontend/src/router.js
import { createRouter, createWebHistory } from 'vue-router'
import Home from './pages/Home.vue'
import { defineAsyncComponent } from 'vue'

// Lazy load the RepoChat route so its JS chunk only loads on /repo
const RepoChat = defineAsyncComponent({
  loader: () => import('./pages/RepoChat.vue'),
  timeout: 20000,
  onError(err, _retry, _fail, attempts) {
    // After one attempt, bubble to the page error boundary
    if (attempts >= 1) throw err
  },
})

export default createRouter({
  history: createWebHistory('/app/'),
  routes: [
    { path: '/', name: 'home', component: Home },
    { path: '/repo', name: 'repo', component: RepoChat },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})
