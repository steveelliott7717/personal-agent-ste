// src/firebase.js
import { initializeApp } from 'firebase/app'

let analytics = null

async function maybeInitAnalytics(app) {
  try {
    const { isSupported, getAnalytics } = await import('firebase/analytics')
    if (await isSupported()) {
      analytics = getAnalytics(app)
    }
  } catch {
    // Analytics not enabled or not supported — ignore
  }
}

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  ...(import.meta.env.VITE_FIREBASE_MEASUREMENT_ID
    ? { measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID }
    : {}),
}

export const firebaseApp = initializeApp(firebaseConfig)
maybeInitAnalytics(firebaseApp)

export { analytics }
