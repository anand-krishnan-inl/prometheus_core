# File Upload Workflow

> **Note:** The Chainlit frontend has been removed; uploads now originate from the React
> `frontend/` and flow through the FastAPI `backend/`. The client-agent → DeepLynx portion below
> is unchanged, but the "Frontend encoding" details referencing `frontend/components/message_handler.py`
> describe the removed Chainlit frontend and are retained only as historical reference.

Users can attach files to a query in the frontend. This document describes how those files flow through the system to DeepLynx and how they become available to specialist agents.

## High-level flow

```
User attaches file(s) in Chainlit
  → frontend/components/message_handler.py optionally renders local viewers
  → frontend streams bytes to client-agent /uploads/stage
  → ConversationRequest.uploaded_files: [{name, mime_type, size_bytes, upload_id}]
  → client-agent/_execute() calls _handle_uploaded_files() (before routing)
      → infer_file_descriptions() — one LLM call, always returns a string per file
      → upload_files_to_deeplynx() — resolves staged file, calls send_deeplynx_artifact()
      → returns ArtifactReference entries for successfully-uploaded files
      → _handle_uploaded_files() merges them into the artifact catalog and returns their Prometheus artifact IDs
  → artifact IDs stored in ConversationRequest.newly_uploaded_artifact_ids
  → chat() injects them into previous_context["newly_uploaded_artifact_ids"]
  → planner._build_prompt() prepends "[User-uploaded <type> for this request]" to their descriptions
  → resolve_artifact_inputs() auto-binds (conservative rules below)
  → downstream skill calls receive the conversation artifact catalog, including DeepLynx metadata
```

## Frontend encoding

`frontend/components/message_handler.py` streams each attached file to the client-agent's `/uploads/stage` endpoint, then includes the returned staged upload ID in the `uploaded_files` payload sent through A2A.  Files handled by a visualization plugin (e.g. IFC, Exodus, STL, image, or SVG viewers) are processed locally for immediate rendering and **also** staged for DeepLynx upload.  The plugin path must not `continue` before `_capture_uploaded_file()` runs.

The upload is capped at `MAX_UPLOAD_BYTES` (100 MB) per file.  A2A carries only metadata, so large file bytes do not inflate the JSON-RPC request body.  Files are passed as:

```json
{
  "name": "reactor_core.exo",
  "mime_type": "application/x-hdf5",
  "size_bytes": 4194304,
  "upload_id": "9f3f5b5d2e704a82b8e99f448b914f7f"
}
```

## Upload plugin inspection

Visualization plugins own upload detection through `inspect_upload(context) -> UploadMatch | None`.  The context includes the temporary file path, display name, initial MIME type, and file size.  Each plugin can combine MIME, extension, and lightweight content inspection, then returns a MIME type and confidence.  The registry chooses the highest-confidence match with deterministic tie-breaking and calls `process_upload(context)` with one structured argument.

This keeps storage independent from viewing: plugin rendering is best-effort local preview, while staging for DeepLynx runs for every readable upload.

## Description inference (`infer_file_descriptions`)

A single LLM call returns a JSON mapping `{filename: description}`.  The prompt instructs the model to use the user's query as context when it's informative, or synthesize a description from the filename and MIME type alone.  The model is explicitly told to **never** return null — a non-empty string is always produced.

If the LLM response cannot be parsed, `_fallback_description()` is used: `"User-uploaded <label>: <filename>"`.

Every description, whether from the LLM or the fallback, is at most 250 characters (`MAX_DESCRIPTION_CHARS`). The prompt
states the limit, and `_limit_description()` enforces it, cutting with an ellipsis when needed. DeepLynx
(deeplynx-server v0.8.0) rejects a longer record description with HTTP 400. That fails the record update
after the file upload, and the file is dropped with only `[WARN] Failed to upload ... HTTP 400` in the
client-agent log.

## MIME-to-label mapping (`mime_type_label`)

`file_upload_handler.mime_type_label(mime_type, filename)` produces a human-readable label such as `"Exodus mesh file"` or `"MOOSE input file"`.  It checks:

1. Exact MIME matches (`_EXACT_MIME_LABELS`)
2. File extension matches (`_EXT_LABELS`) — covers nuclear-engineering formats: `.exo`, `.e`, `.i`, `.inp`, `.ifc`, `.h5`, `.nc`, `.vtu`, `.step`, etc.
3. MIME prefix fallbacks (`image/`, `video/`, `audio/`, `text/`, `application/`)

## DeepLynx upload

`upload_files_to_deeplynx` resolves each staged upload file, validates it against the metadata sent through A2A, calls `agent.send_deeplynx_artifact()`, then consumes the staged file.  Each uploaded file becomes a conversation-scoped `ArtifactReference` with:

- `tags: ["user-upload"]`
- `metadata: {"original_filename": "<name>", "source": "user-upload", ...}` plus storage metadata added by `send_deeplynx_artifact()` such as `deeplynx_file_id`
- The inferred description stored in DeepLynx

The temporary file is cleaned up regardless of upload success or failure.

`upload_files_to_deeplynx()` handles individual file failures by logging a warning and continuing with the remaining files.  Invalid staged metadata, missing staged bytes, size mismatches, and MIME mismatches are rejected before storage.

## Artifact IDs vs DeepLynx IDs

The system has two different identifier domains:

- **Prometheus `artifact_id`**: the UUID stored in `ArtifactReference.artifact_id`.  This is the ID shown in the conversation artifact catalog and the ID that `x_artifact_ref` skill parameters must receive.
- **DeepLynx identifiers**: DeepLynx item/file identifiers shown in DeepLynx UI fields such as "Original ID" or stored in artifact metadata such as `deeplynx_file_id`.  These are storage-layer identifiers and should not be passed as `x_artifact_ref` values.

For follow-up turns, the frontend sends the current conversation artifact catalog back to the client-agent.  Downstream skills should pass the Prometheus `artifact_id` to `PrometheusAgent.get_artifact_file_path()`, along with any orchestrator-provided `params["artifacts"]` catalog, to receive a readable local file path.

## Ephemeral upload signal in the planner

After upload, the new Prometheus artifact IDs are stored as `newly_uploaded_artifact_ids` in the conversation context **for the current turn only**.  `merge_artifacts_into_context` drops this key at the start of each subsequent turn so stale data never propagates.

In `planner._build_prompt()`, artifacts whose IDs appear in `newly_uploaded_artifact_ids` have their descriptions prefixed with:

```
[User-uploaded <label> for this request]
```

This prefix helps the planner match the uploaded file to the correct skill parameter even when the user's query uses natural language (e.g. "Convert the attached image").

Example artifact block entry:

```
- reactor_core.exo (id: abc-123) [User-uploaded Exodus mesh file for this request] Reactor core Exodus mesh for neutron flux analysis.
```

## Conservative auto-binding (`resolve_artifact_inputs`)

After the standard UUID/name resolution pass, `resolve_artifact_inputs` performs a conservative auto-binding step:

**Condition**: exactly 1 valid newly-uploaded artifact ID AND exactly 1 still-unfilled `x_artifact_ref` field in the skill's input schema.

**Action**: bind that artifact ID to the field (single-value or list depending on `x_artifact_ref` mode).

**Otherwise**: leave the field empty; the planner is responsible for explicit binding via `set_extracted_params`.

This handles the common single-file upload case without overriding explicit matches or guessing when there is ambiguity.

## Downstream specialist access

Before invoking specialist agents, the client-agent calls `attach_artifact_context()` to add the conversation artifact catalog to the skill params when artifacts are available.  Workflow-template execution also carries `context["artifacts"]` into each step payload when the step does not set an explicit `artifacts` field.

This is required because a specialist agent's local artifact registry may not contain a file that was uploaded by the client-agent.  Artifact-aware specialists should use `get_artifact_file_path()`; the helper checks the local registry, falls back to the orchestrator-provided `params["artifacts"]` catalog, caches the reference when possible, and returns a local file path.

Example: `artifact-debug-agent`'s SVG-to-PNG converter receives `artifact_id=<Prometheus artifact_id>` and `artifacts=[...]`, calls `get_artifact_file_path()`, and converts the returned SVG path with CairoSVG.  It should not use the DeepLynx "Original ID" as the skill input.

## Multi-artifact turns

When the user uploads a file **and** references an earlier conversation artifact in the same query (e.g. "Run this input with the mesh you generated earlier"), both artifacts appear in the unified artifact list.  The newly-uploaded file is distinguished only by the ephemeral description prefix.  The planner reads the full context and selects the appropriate artifact IDs for each skill parameter via `set_extracted_params`.

## Error handling

- If DeepLynx is not configured (`ConnectionConfig.from_env()` raises), the upload is skipped with a warning/status update and the artifact IDs list is empty.
- If an individual file upload fails, the warning is logged and the remaining files continue to upload.
- Description inference failures fall back to `"User-uploaded <label>: <filename>"`.
- If files were received but not stored, `_handle_uploaded_files()` adds transient `received_files` summaries to the request context so the planner can accurately say the files were received but are not available as stored artifacts.
- `merge_artifacts_into_context()` drops both `newly_uploaded_artifact_ids` and `received_files` at the start of later turns.
- The router never blocks on upload failures — the request continues without the artifact.
