'use client'

/**
 * Download / open controls for the project report.
 *
 * The PDF is rendered server-side by a locally installed Chrome or Edge, so
 * this really does hand back a `.pdf` file rather than opening a print dialog
 * and hoping. The first render of a large report takes a couple of seconds,
 * hence the loading state; afterwards it is a cached file read.
 *
 * The download goes through `fetch` rather than a plain link so a failure
 * (no browser installed, report not generated) can be shown in place. A bare
 * anchor would navigate the user to a page of error text.
 */
import { AlertTriangle, Download, FileText } from 'lucide-react'
import { useCallback, useState } from 'react'

import { Button, buttonVariants } from '@/components/ui'

export function ReportActions({
  src,
  pdfHref,
  fileName,
  pdfFileName,
}: {
  src: string
  pdfHref: string
  fileName: string
  pdfFileName: string
}) {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const downloadPdf = useCallback(async () => {
    setPending(true)
    setError(null)
    try {
      const response = await fetch(pdfHref)
      if (!response.ok) {
        // The route answers 409 with a sentence meant for a person.
        setError((await response.text()) || `The PDF could not be generated (${response.status}).`)
        return
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = pdfFileName
      document.body.appendChild(link)
      link.click()
      link.remove()
      // Revoking immediately can cancel the download in some browsers.
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
    } catch (cause) {
      setError(
        `Could not reach the workbench: ${cause instanceof Error ? cause.message : String(cause)}`,
      )
    } finally {
      setPending(false)
    }
  }, [pdfHref, pdfFileName])

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          icon={<Download className="size-4" />}
          loading={pending}
          onClick={downloadPdf}
        >
          {pending ? 'Rendering PDF…' : 'Download PDF'}
        </Button>

        <a
          href={`${src}?download`}
          download={fileName}
          className={buttonVariants({ variant: 'secondary' })}
        >
          <Download className="size-4" />
          Download HTML
        </a>

        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className={buttonVariants({ variant: 'ghost' })}
        >
          <FileText className="size-4" />
          Open in a new tab
        </a>
      </div>

      {pending && (
        <p className="text-caption text-text-secondary" role="status">
          Rendering with your local browser engine. A large report takes a few seconds; the result
          is cached for next time.
        </p>
      )}

      {error && (
        <p
          role="alert"
          className="flex items-start gap-2 rounded-md border border-danger bg-danger-subtle px-3 py-2 text-caption text-danger"
        >
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span>{error}</span>
        </p>
      )}
    </div>
  )
}

/** One row per downloadable deliverable: what it is, where it lives, a button. */
export function DeliverableRow({
  label,
  path,
  href,
  available,
  fileName,
}: {
  label: string
  path: string
  href: string
  available: boolean
  fileName: string
}) {
  return (
    <div className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <p className="text-body text-text">{label}</p>
        <code className="mt-1 block truncate text-label text-text-tertiary">{path}</code>
      </div>
      {available ? (
        <a
          href={`${href}?download`}
          download={fileName}
          className={`shrink-0 ${buttonVariants({ variant: 'secondary', size: 'sm' })}`}
          aria-label={`Download ${label}`}
        >
          <Download className="size-4" />
          Download
        </a>
      ) : (
        <span className="shrink-0 text-caption text-text-tertiary">not generated yet</span>
      )}
    </div>
  )
}
