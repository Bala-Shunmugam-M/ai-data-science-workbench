import { PageHeader } from '@/components/shell/page-header'
import { Card, ThemeToggle } from '@/components/ui'
import { MAX_UPLOAD_BYTES } from '@/lib/uploads'
import { formatBytes } from '@/lib/format'
import { PROJECT_ROOT, WORKSPACES_DIR } from '@/lib/workspace'
import { PYTHON } from '@/lib/python'

export const dynamic = 'force-dynamic'

function Row({
  label,
  help,
  children,
}: {
  label: string
  help?: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-2 py-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="min-w-0">
        <p className="text-body text-text">{label}</p>
        {help && <p className="mt-1 text-caption text-text-secondary">{help}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

function Value({ children }: { children: React.ReactNode }) {
  return (
    <code className="block max-w-full overflow-x-auto rounded-md border border-border bg-bg-subtle px-3 py-1.5 text-caption text-text-secondary">
      {children}
    </code>
  )
}

export default function SettingsPage() {
  return (
    <>
      <PageHeader
        title="Settings"
        description="This app runs against the Python workbench on this machine. These are the paths and limits it is using."
      />

      <div className="flex max-w-3xl flex-col gap-6">
        <Card className="flex flex-col">
          <h2 className="text-h3 text-text">Appearance</h2>
          <div className="divide-y divide-border">
            <Row label="Theme" help="Follows your operating system unless you choose otherwise.">
              <ThemeToggle />
            </Row>
          </div>
        </Card>

        <Card className="flex flex-col">
          <h2 className="text-h3 text-text">Analysis</h2>
          <div className="divide-y divide-border">
            <Row label="Maximum upload size" help="Larger files are rejected before any work starts.">
              <Value>{formatBytes(MAX_UPLOAD_BYTES)}</Value>
            </Row>
            <Row
              label="Preview rows"
              help="How many rows the results and preview screens show. The model always uses every row."
            >
              <Value>50</Value>
            </Row>
            <Row label="Python interpreter" help="Override with the WORKBENCH_PYTHON environment variable.">
              <Value>{PYTHON}</Value>
            </Row>
          </div>
        </Card>

        <Card className="flex flex-col">
          <h2 className="text-h3 text-text">Storage</h2>
          <p className="mt-1 text-caption text-text-secondary">
            Nothing is uploaded anywhere. Every dataset and every artifact stays in these folders on
            this computer until you delete them yourself.
          </p>
          <div className="mt-2 divide-y divide-border">
            <Row label="Workbench root">
              <Value>{PROJECT_ROOT}</Value>
            </Row>
            <Row label="Project workspaces">
              <Value>{WORKSPACES_DIR}</Value>
            </Row>
          </div>
        </Card>
      </div>
    </>
  )
}
