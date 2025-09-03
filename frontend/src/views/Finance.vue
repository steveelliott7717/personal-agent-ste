<template>
  <div>
    <h1>Workouts</h1>
    <input v-model="workout" placeholder="Log a workout" />
    <button @click="logWorkout">Log Workout</button>
    <ul>
      <li v-for="w in workouts" :key="w.id">{{ w.description }}</li>
    </ul>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return { workout: '', workouts: [] }
  },
  mounted() {
    this.loadWorkouts()
  },
  methods: {
    async logWorkout() {
      await axios.post(
        '/api/request',
        new URLSearchParams({ query: `log workout ${this.workout}` })
      )
      this.loadWorkouts()
    },
    async loadWorkouts() {
      const res = await axios.post(
        '/api/request',
        new URLSearchParams({ query: 'show workouts' })
      )
      this.workouts = res.data.result
    },
  },
}
</script>
