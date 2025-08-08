<template>
  <div>
    <h1>Meals</h1>
    <input v-model="meal" placeholder="Log a meal">
    <button @click="logMeal">Log Meal</button>
    <ul>
      <li v-for="m in meals" :key="m.id">{{ m.description }}</li>
    </ul>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return { meal: '', meals: [] }
  },
  methods: {
    async logMeal() {
      await axios.post('/api/request', new URLSearchParams({ query: `log meal ${this.meal}` }))
      this.loadMeals()
    },
    async loadMeals() {
      const res = await axios.post('/api/request', new URLSearchParams({ query: 'show meals' }))
      this.meals = res.data.result
    }
  },
  mounted() { this.loadMeals() }
}
</script>
