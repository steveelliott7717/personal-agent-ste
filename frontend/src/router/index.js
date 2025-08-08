import { createRouter, createWebHistory } from 'vue-router'

import Dashboard from '../views/Dashboard.vue'
import Meals from '../views/Meals.vue'
import Workouts from '../views/Workouts.vue'
import Finance from '../views/Finance.vue'
import Grooming from '../views/Grooming.vue'
import Notifications from '../views/Notifications.vue'
import AuditLog from '../views/AuditLog.vue'
import Settings from '../views/Settings.vue'

const routes = [
  { path: '/', component: Dashboard },
  { path: '/meals', component: Meals },
  { path: '/workouts', component: Workouts },
  { path: '/finance', component: Finance },
  { path: '/grooming', component: Grooming },
  { path: '/notifications', component: Notifications },
  { path: '/audit', component: AuditLog },
  { path: '/settings', component: Settings }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
