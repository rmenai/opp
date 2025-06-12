// @ts-check
import withNuxt from "./.nuxt/eslint.config.mjs";
import eslintPluginPrettierRecommended from "eslint-plugin-prettier/recommended";

export default withNuxt(
  eslintPluginPrettierRecommended,

  {
    files: ["**/*.vue"],
    rules: {
      "vue/multi-word-component-names": "off",
      "vue/no-multiple-template-root": "off",
    },
  },
);
