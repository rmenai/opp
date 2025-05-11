import tailwindcss from "@tailwindcss/vite";

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2024-11-01",
  future: { compatibilityVersion: 4 },
  devtools: { enabled: true },

  css: ["./app/assets/css/main.css"],

  vite: {
    plugins: [tailwindcss()],
  },

  modules: [
    "@nuxt/eslint",
    "@nuxt/fonts",
    "@nuxt/icon",
    "@nuxt/test-utils",
    "@nuxt/ui",
    "@nuxt/image",
    "@pinia/nuxt",
    "@nuxt/test-utils/module",
    "@vueuse/nuxt",
  ],
});