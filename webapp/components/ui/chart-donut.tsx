'use client'

import { Cell, Legend, Pie, PieChart, Tooltip } from 'recharts'
import {
  ChartFrame,
  LEGEND_STYLE,
  TOOLTIP_STYLE,
  seriesColor,
  type ChartDatum,
  type ChartSeries,
} from './chart-common'

export interface ChartDonutProps {
  data: ChartDatum[]
  /** Key holding each slice's label. */
  xKey: string
  /**
   * One series — the numeric key to size slices by. Extra entries are
   * ignored (a donut can only encode one measure).
   * Keep `data` to 6 slices or fewer; beyond that the 6-colour ramp repeats.
   */
  series: ChartSeries[]
  /** Required plain-language line rendered under the chart. */
  caption: string
  height?: number
  className?: string
}

export function ChartDonut({
  data,
  xKey,
  series,
  caption,
  height = 260,
  className,
}: ChartDonutProps) {
  const value = series[0]
  return (
    <ChartFrame
      caption={caption}
      height={height}
      labelKey={xKey}
      labelHeader={xKey}
      series={value ? [value] : []}
      data={data}
      className={className}
    >
      <PieChart margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
        <Pie
          data={data}
          dataKey={value?.key ?? ''}
          nameKey={xKey}
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
          stroke="var(--surface)"
          strokeWidth={2}
          isAnimationActive={false}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={seriesColor(i)} />
          ))}
        </Pie>
        <Tooltip {...TOOLTIP_STYLE} cursor={false} />
        <Legend wrapperStyle={LEGEND_STYLE} verticalAlign="bottom" height={28} />
      </PieChart>
    </ChartFrame>
  )
}
