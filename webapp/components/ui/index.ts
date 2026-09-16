// Barrel for the P3 component library: import { Button, Card } from '@/components/ui'

// Use THIS cn, not lib/utils' — see cn.ts. lib/utils' twMerge silently drops
// the custom type-scale classes (text-metric, text-h1, text-caption, …) when
// a text colour is merged alongside them.
export { cn } from './cn'

export { Button, type ButtonProps } from './button'
// From the boundary-free module, so Server Components can call it to style a
// <Link> as a button. Re-exporting it from './button' would drag the client
// boundary along with it.
export { buttonVariants, type ButtonVariantProps } from './button-variants'
export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  type CardProps,
} from './card'
export { Badge, badgeVariants, type BadgeProps } from './badge'
export { MetricTile, type MetricTileProps, type MetricTone } from './metric-tile'
export {
  DataTable,
  type DataTableProps,
  type DataTableColumn,
  type DataTableRow,
} from './data-table'
export { EmptyState, type EmptyStateProps } from './empty-state'
export { Skeleton, type SkeletonProps } from './skeleton'
export {
  ProgressSteps,
  type ProgressStepsProps,
  type ProgressStep,
  type StepState,
} from './progress-steps'
export {
  DropZone,
  type DropZoneProps,
  type DropZoneSample,
} from './drop-zone'
export { Toast, ToastRegion, type ToastProps, type ToastTone } from './toast'
export {
  CHART_COLORS,
  MAX_SERIES,
  seriesColor,
  type ChartSeries,
  type ChartDatum,
} from './chart-common'
export { ChartBar, type ChartBarProps } from './chart-bar'
export { ChartLine, type ChartLineProps } from './chart-line'
export { ChartDonut, type ChartDonutProps } from './chart-donut'
export { Tabs, type TabItem } from './tabs'
export { ThemeToggle } from './theme-toggle'
