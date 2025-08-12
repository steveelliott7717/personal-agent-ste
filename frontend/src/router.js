// frontend/src/router.js
import { createRouter, createWebHistory } from 'vue-router'
import Home from './pages/Home.vue'
import RepoChat from './pages/RepoChat.vue'

export default createRouter({
  history: createWebHistory('/app/'),
  routes: [
    { path: '/', component: Home, name: 'home' },
    { path: '/repo', component: RepoChat, name: 'repo' },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})
