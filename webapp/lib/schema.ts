/**
 * The human-approved semantic schema, fetched once from Python.
 *
 * `SEMANTIC_SCHEMA` is a Python constant with no on-disk representation, so the
 * only way to show it is to ask the bridge. That costs a subprocess and a
 * pandas import (a few seconds), which is far too slow to repeat on every Data
 * page render — hence the module-level promise cache. The schema cannot change
 * while the server is running, so caching for the process lifetime is exactly
 * as fresh as reading it every time.
 */
import 'server-only'

import { runBridgeJson } from '@/lib/python'

export interface SchemaRow {
  column: string
  role: string | null
  semantic_type: string | null
  expected_storage_type: string | null
  nullable: boolean | null
  minimum: number | null
  maximum: number | null
  whole_number: boolean | null
  preprocessing_group: string | null
  missing_treatment: string | null
  allowed_values: string[] | null
}

let cached: Promise<SchemaRow[]> | null = null

export function semanticSchema(): Promise<SchemaRow[]> {
  if (!cached) {
    cached = runBridgeJson<{ schema: SchemaRow[] }>(['schema'], 60_000)
      .then((payload) => payload.schema)
      // A failure must not take the Data page down with it: the schema is one
      // section of it. Reset the cache so a later render can retry.
      .catch(() => {
        cached = null
        return []
      })
  }
  return cached
}
