// frontend/src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import('../pages/Home.vue'),
  },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: () => import('../views/Dashboard.vue'),
  },
  {
    path: '/repo-chat',
    name: 'repo-chat',
    component: () => import('../pages/RepoChat.vue'),
  },
  // 404 fallback: keep last
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: {
      template:
        "<main style='padding:2rem;font:14px/1.4 system-ui'>404 — page not found</main>",
    },
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
