import { NextResponse } from 'next/server'

import { projectSummaries } from '@/lib/workspace'

// Always read the disk: a project can appear while the app is running.
export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    return NextResponse.json(await projectSummaries())
  } catch (error) {
    return NextResponse.json(
      {
        error: 'Could not read the workspaces directory',
        hint: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    )
  }
}
