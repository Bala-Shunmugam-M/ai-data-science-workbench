/**
 * Re-export so `@/components/ui` and `@/lib/utils` cannot diverge.
 *
 * The type-scale-aware tailwind-merge config lives in `lib/utils.ts` — see the
 * comment there for why stock `twMerge` drops `text-metric` and friends.
 */
export { cn } from '@/lib/utils'
