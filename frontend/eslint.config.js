// eslint.config.js
import js from '@eslint/js'
import vue from 'eslint-plugin-vue'
import prettier from 'eslint-config-prettier'
import globals from 'globals'

export default [
  // Ignore build + deps
  {
    ignores: ['dist/**', 'node_modules/**'],
  },

  // Base configs
  js.configs.recommended,
  ...vue.configs['flat/recommended'],
  prettier,

  // Application source (browser)
  {
    files: ['src/**/*.{js,vue}', 'public/**/*.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        ...globals.browser, // window, document, fetch, localStorage, etc.
      },
    },
    rules: {
      'no-unused-vars': 'warn',
      'no-prototype-builtins': 'off',
      'vue/multi-word-component-names': 'off',
      'vue/attributes-order': 'warn',
      'vue/order-in-components': 'warn',
    },
  },

  // Service workers
  {
    files: ['public/service-worker.js', 'src/**/service-worker.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        ...globals.serviceworker, // self, caches, clients, etc.
      },
    },
  },

  // Node-based config files
  {
    files: ['vite.config.{js,ts}', 'eslint.config.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        ...globals.node,
      },
    },
    rules: {
      'no-console': 'off',
    },
  },
]
