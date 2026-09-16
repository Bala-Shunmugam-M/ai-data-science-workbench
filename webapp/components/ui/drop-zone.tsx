'use client'

import { AlertCircle, UploadCloud } from 'lucide-react'
import { useId, useRef, useState, type DragEvent } from 'react'
import { cn } from './cn'
import { formatBytes } from '@/lib/format'

export interface DropZoneSample {
  /** Shown on the chip. */
  name: string
  /** Passed back to `onSampleSelect`; defaults to `name`. */
  id?: string
  hint?: string
}

export interface DropZoneProps {
  onFile: (file: File) => void
  /** Comma-separated extensions and/or MIME types, e.g. '.csv,text/csv'. */
  accept?: string
  maxBytes?: number
  sampleFiles?: DropZoneSample[]
  onSampleSelect?: (id: string) => void
  /** Rejections are also surfaced inline; this is for logging/toasts. */
  onError?: (message: string) => void
  /** Overrides the internal rejection message (e.g. a server-side error). */
  error?: string | null
  disabled?: boolean
  className?: string
}

function accepts(file: File, accept: string): boolean {
  const name = file.name.toLowerCase()
  const type = file.type.toLowerCase()
  return accept
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean)
    .some((rule) =>
      rule.startsWith('.')
        ? name.endsWith(rule)
        : rule.endsWith('/*')
          ? type.startsWith(rule.slice(0, -1))
          : type === rule,
    )
}

export function DropZone({
  onFile,
  accept = '.csv',
  maxBytes,
  sampleFiles,
  onSampleSelect,
  onError,
  error,
  disabled,
  className,
}: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [rejection, setRejection] = useState<string | null>(null)
  const formatsId = useId()
  const errorId = useId()

  const message = error ?? rejection
  const formats = `${accept.toUpperCase().replace(/\./g, '')} files${
    maxBytes ? ` up to ${formatBytes(maxBytes, 0)}` : ''
  }`

  function reject(reason: string) {
    setRejection(reason)
    onError?.(reason)
  }

  function handle(file: File | undefined) {
    if (!file) return
    if (accept && !accepts(file, accept)) {
      return reject(
        `“${file.name}” isn’t an accepted file type. This upload takes ${formats}.`,
      )
    }
    if (maxBytes && file.size > maxBytes) {
      return reject(
        `“${file.name}” is ${formatBytes(file.size)} — the limit is ${formatBytes(maxBytes, 0)}.`,
      )
    }
    setRejection(null)
    onFile(file)
  }

  function onDrop(e: DragEvent<HTMLButtonElement>) {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    handle(e.dataTransfer.files?.[0])
  }

  return (
    <div className={cn('flex flex-col gap-3', className)}>
      <button
        type="button"
        disabled={disabled}
        aria-describedby={message ? `${formatsId} ${errorId}` : formatsId}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          'flex min-h-[280px] w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center',
          'transition-colors duration-base ease-standard',
          'outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
          'disabled:cursor-not-allowed disabled:opacity-50',
          dragging
            ? 'border-brand bg-brand-subtle'
            : message
              ? 'border-danger bg-danger-subtle'
              : 'border-border-strong bg-bg-subtle hover:border-brand hover:bg-surface-hover',
        )}
      >
        <UploadCloud
          aria-hidden
          className={cn(
            'size-10 stroke-[1.5]',
            dragging ? 'text-brand' : 'text-text-tertiary',
          )}
        />
        <span className="text-body-lg text-text">
          Drag &amp; drop your CSV here or click to browse
        </span>
        <span id={formatsId} className="text-caption text-text-tertiary">
          {formats}
        </span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          handle(e.target.files?.[0])
          // Allow re-picking the same file.
          e.target.value = ''
        }}
      />

      {message && (
        <p
          id={errorId}
          role="alert"
          className="flex items-start gap-2 text-caption text-danger"
        >
          <AlertCircle aria-hidden className="mt-0.5 size-4 shrink-0" />
          {message}
        </p>
      )}

      {sampleFiles && sampleFiles.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-caption text-text-secondary">
            Or try a sample:
          </span>
          {sampleFiles.map((sample) => (
            <button
              key={sample.id ?? sample.name}
              type="button"
              disabled={disabled}
              title={sample.hint}
              onClick={() => onSampleSelect?.(sample.id ?? sample.name)}
              className={cn(
                'rounded-full border border-border-strong bg-surface px-3 py-1 text-caption text-text-secondary',
                'transition-colors duration-fast ease-standard',
                'hover:border-brand hover:bg-brand-subtle hover:text-brand',
                'active:bg-surface-hover',
                'outline-none focus-visible:ring-2 focus-visible:ring-focus-ring focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
                'disabled:pointer-events-none disabled:opacity-50',
              )}
            >
              {sample.name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
