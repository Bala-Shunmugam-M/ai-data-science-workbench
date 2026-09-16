// Pure, dependency-free formatters shared across the app.
// Locale is pinned to 'en-US' everywhere so numbers/dates render
// identically on the server and the client (no hydration mismatch)
// regardless of the host machine's OS locale.

const LOCALE = 'en-US'

/** Thousands-separated number, e.g. 1234567 -> "1,234,567". */
export function formatNumber(value: number, maxFractionDigits = 0): string {
  if (!Number.isFinite(value)) return '—'
  return new Intl.NumberFormat(LOCALE, {
    maximumFractionDigits: maxFractionDigits,
  }).format(value)
}

/** Byte count -> human size, e.g. 1536 -> "1.5 KB". Binary (1024) units. */
export function formatBytes(bytes: number, decimals = 1): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  )
  const value = bytes / Math.pow(1024, exponent)
  const fixed = exponent === 0 ? value.toFixed(0) : value.toFixed(decimals)
  return `${fixed} ${units[exponent]}`
}

/** ISO date/datetime string or Date -> "Jul 31, 2026" (stable, locale-pinned). */
export function formatDate(value: string | Date): string {
  const date = typeof value === 'string' ? new Date(value) : value
  if (Number.isNaN(date.getTime())) return '—'
  return new Intl.DateTimeFormat(LOCALE, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(date)
}

/**
 * Fraction (0..1) or already-a-percent number -> "84.5%".
 * `isFraction` defaults to true (0.845 -> "84.5%"); pass false if the
 * input is already on a 0-100 scale.
 */
export function formatPercent(
  value: number,
  { isFraction = true, decimals = 1 }: { isFraction?: boolean; decimals?: number } = {},
): string {
  if (!Number.isFinite(value)) return '—'
  const pct = isFraction ? value * 100 : value
  return `${pct.toFixed(decimals)}%`
}

/**
 * ML-metric-friendly number formatting: respects significant digits
 * rather than a fixed decimal count, so 0.8448 -> "0.8448" (4 sig figs)
 * and 123.456 -> "123.5" and 1234.5 -> "1,235" (>= 1000 falls back to
 * thousands-separated whole numbers) — never trailing zeros padded
 * on to a small metric, never a wall of digits on a big one.
 */
export function formatMetric(value: number, sigFigs = 4): string {
  if (!Number.isFinite(value)) return '—'
  if (value === 0) return '0'

  const abs = Math.abs(value)

  // Large numbers: thousands separators, no decimals.
  if (abs >= 1000) {
    return formatNumber(value, 0)
  }

  // Otherwise round to `sigFigs` significant digits, then trim any
  // trailing zeros introduced by toPrecision's fixed formatting.
  const precise = Number(value.toPrecision(sigFigs))
  const str = precise.toString()

  // toPrecision can fall back to exponential notation for very small
  // numbers (e.g. 0.00001234) — Number(...).toString() keeps that;
  // guard by using tabular fixed notation in that case instead.
  if (str.includes('e') || str.includes('E')) {
    const decimals = Math.max(0, sigFigs - Math.floor(Math.log10(abs)) - 1)
    return precise.toFixed(decimals)
  }

  return new Intl.NumberFormat(LOCALE, {
    maximumFractionDigits: 10,
  }).format(precise)
}
