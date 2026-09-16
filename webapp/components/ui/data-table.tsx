import type { ReactNode } from 'react'
import { cn } from './cn'

export interface DataTableColumn {
  /** Key into each row object. */
  key: string
  label: string
  /**
   * Right-align + tabular figures. Omit to auto-detect from the first
   * non-empty value in the column.
   */
  numeric?: boolean
  /** Any CSS width for the column, e.g. '12rem'. */
  width?: string
  /**
   * Let long text wrap instead of forcing it onto one line. Omit to
   * auto-detect: a column holding prose (rationales, payloads, business
   * meanings) wraps and is width-capped, a column of short labels does not.
   */
  wrap?: boolean
}

export type DataTableRow = Record<string, ReactNode>

export interface DataTableProps {
  columns: DataTableColumn[]
  rows: DataTableRow[]
  /** Any CSS max-height; enables vertical scroll inside the container. */
  maxHeight?: string | number
  zebra?: boolean
  /** Default true. */
  stickyHeader?: boolean
  /** Shown instead of the table body when `rows` is empty. */
  emptyMessage?: string
  /** Screen-reader description of the table. */
  caption?: string
  className?: string
}

function looksNumeric(value: ReactNode): boolean {
  if (typeof value === 'number') return true
  if (typeof value === 'string' && value.trim() !== '') {
    return Number.isFinite(Number(value.replace(/[,%]/g, '')))
  }
  return false
}

/** Longest string in a column, used to decide whether it holds prose. */
function longestText(rows: DataTableRow[], key: string): number {
  let longest = 0
  for (const row of rows) {
    const value = row[key]
    if (typeof value === 'string' && value.length > longest) longest = value.length
  }
  return longest
}

/**
 * Above this many characters a column is prose, not a label, and forcing it
 * onto one line pushes every other column off-screen.
 */
const PROSE_THRESHOLD = 60

/**
 * Numeric columns are right-aligned with tabular figures. The table scrolls
 * inside its own container so the page body never scrolls horizontally.
 */
export function DataTable({
  columns,
  rows,
  maxHeight,
  zebra = false,
  stickyHeader = true,
  emptyMessage = 'No rows to show.',
  caption,
  className,
}: DataTableProps) {
  // ponytail: one sniff pass here instead of every caller tagging columns.
  const numericByKey = new Map<string, boolean>(
    columns.map((col) => [
      col.key,
      col.numeric ??
        looksNumeric(rows.find((r) => r[col.key] != null && r[col.key] !== '')?.[col.key]),
    ]),
  )

  const wrapByKey = new Map<string, boolean>(
    columns.map((col) => [
      col.key,
      col.wrap ?? (!numericByKey.get(col.key) && longestText(rows, col.key) > PROSE_THRESHOLD),
    ]),
  )

  return (
    <div
      className={cn(
        'overflow-auto rounded-lg border border-border bg-surface',
        className,
      )}
      style={{ maxHeight }}
    >
      {/*
        `border-separate` with zero spacing, NOT `border-collapse`. A sticky
        <th> inside a collapsed-border table does not reliably paint its own
        background — the collapsed border model hands painting to the table —
        so scrolling rows show *through* the header and the text appears to
        collide. Separated borders keep the header opaque; the per-cell
        border-b classes below draw the rules instead.
      */}
      <table className="w-full border-separate border-spacing-0 text-caption">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                style={{ width: col.width }}
                className={cn(
                  // Invariant styling lives in one CSS class (globals.css) so
                  // it is not repeated on every cell of a 7,000-cell table.
                  'dt-head',
                  numericByKey.get(col.key) ? 'text-right' : 'text-left',
                  stickyHeader && 'dt-head-sticky',
                )}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-8 text-center text-caption text-text-secondary"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            rows.map((row, i) => (
              <tr
                key={i}
                className={cn(
                  // In the separated-border model a <tr> cannot draw a border,
                  // so the rule lives on the cells and is removed on the last
                  // row from here.
                  '[&:last-child>td]:border-b-0',
                  zebra && i % 2 === 1 && 'bg-bg-subtle',
                )}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={cn(
                      'dt-cell',
                      numericByKey.get(col.key)
                        ? 'text-right tabular-nums whitespace-nowrap'
                        : 'text-left',
                      wrapByKey.get(col.key) ? 'dt-cell-wrap' : 'whitespace-nowrap',
                    )}
                  >
                    {row[col.key] ?? (
                      <span className="text-text-tertiary">—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}
