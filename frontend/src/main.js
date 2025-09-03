// frontend/src/main.js
import { createApp } from 'vue'
import router from './router'
import App from './App.vue'
import './firebase'

createApp(App).use(router).mount('#app')
