import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      // `npm run verify` builds into `.next-verify` so it cannot clobber a running
      // dev server (next.config.ts, NEXT_DIST_DIR). Unignored, `npm run lint` then
      // reports thousands of problems in generated bundles and the handful that
      // matter in `app/` and `lib/` are invisible.
      ".next-verify/**",
      "out/**",
      "build/**",
      "next-env.d.ts",
    ],
  },
];

export default eslintConfig;
