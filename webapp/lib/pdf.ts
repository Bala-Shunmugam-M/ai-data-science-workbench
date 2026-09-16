/**
 * Server-only: render the project report to a real PDF file.
 *
 * WHY A BROWSER AND NOT A LIBRARY. The report is a self-contained HTML
 * document with its own CSS and base64-embedded figures. Reproducing that
 * faithfully needs a real layout engine, and every option costs something:
 * Puppeteer/Playwright download a ~300 MB Chromium, WeasyPrint needs GTK on
 * Windows, wkhtmltopdf needs its own binary. Meanwhile Windows 11 ships Edge
 * and this machine also has Chrome — both of which print to PDF from the
 * command line. So: no new dependency, real fidelity.
 *
 * The output is cached next to the report and regenerated only when the HTML
 * is newer, because a render costs ~2.3 s for a 2 MB report.
 */
import 'server-only'

import { execFile } from 'node:child_process'
import { access, mkdir, stat } from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'

import { resolveInWorkspace } from '@/lib/artifacts'

const run = promisify(execFile)

/** Candidate browsers, most-likely first. `WORKBENCH_CHROME` overrides all. */
function candidates(): string[] {
  const override = process.env.WORKBENCH_CHROME
  if (override) return [override]

  const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files'
  const programFilesX86 = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)'
  const localAppData = process.env.LOCALAPPDATA ?? ''

  if (process.platform === 'win32') {
    return [
      path.join(programFilesX86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(programFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      ...(localAppData
        ? [path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe')]
        : []),
    ]
  }
  return [
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  ]
}

let cachedBrowser: string | null | undefined

export async function findBrowser(): Promise<string | null> {
  if (cachedBrowser !== undefined) return cachedBrowser
  for (const candidate of candidates()) {
    try {
      await access(candidate)
      cachedBrowser = candidate
      return candidate
    } catch {
      /* try the next one */
    }
  }
  cachedBrowser = null
  return null
}

export class PdfUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'PdfUnavailableError'
  }
}

async function newerThan(a: string, b: string): Promise<boolean> {
  try {
    const [left, right] = await Promise.all([stat(a), stat(b)])
    return left.mtimeMs > right.mtimeMs
  } catch {
    return true
  }
}

/**
 * Absolute path to the report's PDF, rendering it if needed.
 *
 * Throws {@link PdfUnavailableError} when no browser is installed or the
 * source report does not exist — both are states the caller should report
 * plainly rather than retry.
 */
export async function reportPdfPath(slug: string): Promise<string> {
  const html = resolveInWorkspace(slug, 'artifacts/reports/project_report.html')
  const pdf = resolveInWorkspace(slug, 'artifacts/reports/project_report.pdf')
  if (!html || !pdf) throw new PdfUnavailableError('Bad project name.')

  try {
    await access(html)
  } catch {
    throw new PdfUnavailableError(
      'This project has no project_report.html to convert. Run the report stage first.',
    )
  }

  // Reuse unless the report has been regenerated since.
  let needsRender = true
  try {
    await access(pdf)
    needsRender = await newerThan(html, pdf)
  } catch {
    needsRender = true
  }
  if (!needsRender) return pdf

  const browser = await findBrowser()
  if (!browser) {
    throw new PdfUnavailableError(
      'No Chrome or Edge installation was found to render the PDF. Set WORKBENCH_CHROME ' +
        'to a browser executable, or use "Download HTML" and print it yourself.',
    )
  }

  await mkdir(path.dirname(pdf), { recursive: true })
  const fileUrl = `file:///${html.replace(/\\/g, '/')}`

  try {
    await run(
      browser,
      [
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        // Chrome's default adds a URL and page numbers in the margins; the
        // report has its own title block and does not want a file:// path
        // stamped across the top of every page.
        '--no-pdf-header-footer',
        `--print-to-pdf=${pdf}`,
        fileUrl,
      ],
      // A 2 MB report renders in ~2.5 s; the ceiling is for a much larger one.
      { timeout: 120_000, windowsHide: true, maxBuffer: 1024 * 1024 },
    )
  } catch (error) {
    throw new PdfUnavailableError(
      `The browser failed to render the PDF: ${
        error instanceof Error ? error.message.split('\n')[0] : String(error)
      }`,
    )
  }

  try {
    await access(pdf)
  } catch {
    throw new PdfUnavailableError('The browser exited without writing a PDF.')
  }
  return pdf
}
