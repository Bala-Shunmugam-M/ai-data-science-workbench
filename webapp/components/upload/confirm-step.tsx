'use client'

/**
 * The confirmation the whole flow hinges on.
 *
 * The detected target is a *guess* and it is often wrong: `guess_target` in the
 * Python side falls back to the last column, which on the raw housing file
 * picks `ocean_proximity` rather than `median_house_value`. So this is written
 * as an editable suggestion with the reasoning visible, not as a decision the
 * user has to notice is wrong. Getting this one field right is the difference
 * between a useful model and a nonsense one.
 */
import { ChevronDown, Loader2, Sparkles, TriangleAlert } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Badge, Button, Card, DataTable, cn } from '@/components/ui'
import { formatBytes, formatNumber } from '@/lib/format'
import type { DetectResponse, Task } from '@/lib/types'

import type { ConfirmedSetup } from '@/hooks/use-analysis'

function Disclosure({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 rounded-lg px-4 py-3 text-body text-text transition-colors duration-fast hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus-ring"
      >
        <span>{title}</span>
        <ChevronDown
          className={cn('size-4 shrink-0 text-text-tertiary transition-transform duration-fast', open && 'rotate-180')}
        />
      </button>
      {open && <div className="border-t border-border p-4">{children}</div>}
    </div>
  )
}

export function ConfirmStep({
  file,
  detection,
  onAnalyze,
  onCancel,
}: {
  file: { name: string; sizeBytes: number }
  detection: DetectResponse
  onAnalyze: (setup: ConfirmedSetup) => void
  onCancel: () => void
}) {
  const [target, setTarget] = useState(detection.detection.target)
  const [task, setTask] = useState<Task>(detection.detection.task)
  /**
   * The task is a property of the target, so changing the target re-derives it
   * from Python rather than leaving the previous guess in place. Without this,
   * picking a continuous column while the initial guess was categorical trains
   * a classifier on continuous data — or, on a column with many distinct
   * values, fails deep in the pipeline with a stratification error.
   */
  const [derived, setDerived] = useState(detection.detection)
  const [deriving, setDeriving] = useState(false)
  const [deriveError, setDeriveError] = useState<string | null>(null)
  /** True once the user overrides the task by hand; stops us clobbering them. */
  const taskPinned = useRef(false)

  const redetect = useCallback(
    async (nextTarget: string) => {
      setDeriving(true)
      setDeriveError(null)
      try {
        const response = await fetch('/api/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fileId: detection.fileId, target: nextTarget }),
        })
        const payload = await response.json()
        if (!response.ok) {
          setDeriveError(payload.hint ?? payload.error ?? 'Could not re-check that column.')
          return
        }
        setDerived(payload.detection)
        if (!taskPinned.current) setTask(payload.detection.task)
      } catch (error) {
        setDeriveError(error instanceof Error ? error.message : String(error))
      } finally {
        setDeriving(false)
      }
    },
    [detection.fileId],
  )

  useEffect(() => {
    // Skip the initial render: the first detection is already in hand.
    if (target === detection.detection.target) {
      setDerived(detection.detection)
      return
    }
    void redetect(target)
  }, [target, detection.detection, redetect])
  const [displayName, setDisplayName] = useState(() =>
    file.name
      .replace(/\.csv$/i, '')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (character) => character.toUpperCase())
      .slice(0, 60),
  )

  const columns = detection.preview.columns
  const changedTarget = target !== detection.detection.target
  // Compare against the detection for the *currently chosen* target, not the
  // original one, or every changed target looks like a manual task override.
  const changedTask = task !== derived.task

  const profileRows = useMemo(
    () =>
      detection.profile.map((column) => ({
        name: column.name,
        dtype: column.dtype,
        missing: column.missing,
        missingPct: `${column.missingPct.toFixed(1)}%`,
        unique: column.unique,
        example: column.example,
      })),
    [detection.profile],
  )

  const previewRows = useMemo(
    () =>
      detection.preview.rows.slice(0, 20).map((row) =>
        Object.fromEntries(columns.map((column, index) => [column, row[index] ?? ''])),
      ),
    [detection.preview.rows, columns],
  )

  const positiveClass =
    task === 'classification' && task === derived.task ? derived.positiveClass : null

  return (
    <div className="flex flex-col gap-6">
      <Card className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h2 className="text-h3 text-text">{file.name}</h2>
          <Badge tone="good">Scanned</Badge>
        </div>
        <p className="text-caption text-text-secondary">
          {formatNumber(detection.rows)} rows · {detection.columns} columns ·{' '}
          {formatBytes(file.sizeBytes)}
        </p>
      </Card>

      <Card className="flex flex-col gap-6">
        <div className="flex items-start gap-3">
          <Sparkles className="mt-0.5 size-5 shrink-0 text-brand" aria-hidden />
          <div>
            <h2 className="text-h3 text-text">Confirm what to predict</h2>
            <p className="mt-1 text-body text-text-secondary">
              These are the workbench&rsquo;s best guesses from the column shapes. Check the target
              before running &mdash; it is the one thing worth two seconds of your attention.
            </p>
          </div>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <div>
            <label htmlFor="target" className="text-label uppercase tracking-wide text-text-tertiary">
              Target column
            </label>
            <select
              id="target"
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              className="mt-2 w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-body text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
            >
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
            <p className="mt-2 text-caption text-text-tertiary">
              {changedTarget
                ? `Changed from the suggested "${detection.detection.target}".`
                : 'This is the column the model will learn to predict.'}
            </p>
          </div>

          <div>
            <span className="text-label uppercase tracking-wide text-text-tertiary">Task</span>
            <div
              role="radiogroup"
              aria-label="Task"
              className="mt-2 inline-flex rounded-md border border-border-strong p-0.5"
            >
              {(['regression', 'classification'] as Task[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  role="radio"
                  aria-checked={task === option}
                  onClick={() => {
                    taskPinned.current = true
                    setTask(option)
                  }}
                  className={cn(
                    'rounded px-3 py-1.5 text-caption capitalize transition-colors duration-fast',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring',
                    task === option
                      ? 'bg-brand text-brand-fg'
                      : 'text-text-secondary hover:text-text',
                  )}
                >
                  {option}
                </button>
              ))}
            </div>
            <p className="mt-2 flex items-center gap-1.5 text-caption text-text-tertiary">
              {deriving && <Loader2 aria-hidden className="size-3 animate-spin" />}
              {deriving
                ? `Re-checking "${target}"…`
                : changedTask
                  ? `Overriding the detected "${derived.task}".`
                  : task === 'classification'
                    ? `Detected ${derived.nClasses ?? '?'} distinct values${positiveClass ? `, scoring "${positiveClass}" as positive` : ''}.`
                    : 'The target is continuous, so this predicts a number.'}
            </p>
          </div>

          <div className="md:col-span-2">
            <label htmlFor="name" className="text-label uppercase tracking-wide text-text-tertiary">
              Project name
            </label>
            <input
              id="name"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              maxLength={60}
              className="mt-2 w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-body text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
            />
          </div>
        </div>

        {deriveError && (
          <p className="flex items-start gap-2 rounded-md border border-danger bg-danger-subtle px-3 py-2 text-caption text-danger">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
            <span>{deriveError}</span>
          </p>
        )}

        {derived.warnings.length > 0 && (
          <ul className="flex flex-col gap-2">
            {derived.warnings.map((warning) => (
              <li
                key={warning}
                className="flex items-start gap-2 rounded-md border border-warning bg-warning-subtle px-3 py-2 text-caption text-warning"
              >
                <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        )}

        {derived.dropColumns.length > 0 && (
          <p className="text-caption text-text-secondary">
            Ignoring <span className="text-text">{derived.dropColumns.join(', ')}</span> &mdash;
            these look like identifiers rather than predictors.
          </p>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="lg"
            onClick={() =>
              onAnalyze({
                displayName: displayName.trim() || file.name.replace(/\.csv$/i, ''),
                target,
                task,
                positiveClass,
              })
            }
            loading={deriving}
            disabled={!displayName.trim() || deriving}
          >
            Upload &amp; Analyze
          </Button>
          <Button variant="ghost" size="lg" onClick={onCancel}>
            Choose a different file
          </Button>
        </div>
      </Card>

      <div className="flex flex-col gap-3">
        <Disclosure title={`Preview (first ${previewRows.length} rows)`}>
          <DataTable
            columns={columns.map((column) => ({ key: column, label: column }))}
            rows={previewRows}
            maxHeight={320}
          />
        </Disclosure>
        <Disclosure title={`Column profile (${detection.profile.length} columns)`}>
          <DataTable
            columns={[
              { key: 'name', label: 'Column' },
              { key: 'dtype', label: 'Type' },
              { key: 'missing', label: 'Missing', numeric: true },
              { key: 'missingPct', label: '% missing', numeric: true },
              { key: 'unique', label: 'Distinct', numeric: true },
              { key: 'example', label: 'Examples' },
            ]}
            rows={profileRows}
            maxHeight={320}
          />
        </Disclosure>
      </div>
    </div>
  )
}
