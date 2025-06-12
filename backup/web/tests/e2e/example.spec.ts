import { test } from "@nuxt/test-utils/playwright";

test("homepage loads correctly", async ({ page }) => {
  await page.goto("/");
});
