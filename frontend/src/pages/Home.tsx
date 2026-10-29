import { useCallback, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchLlmSettings,
  ingest,
  status,
  streamSearch,
  updateLlmSettings,
  type LlmSettings,
  type SearchData,
  type StreamEvent,
  type NodeStartData,
  type NodeResultData,
  type StatusData,
} from '../api';
import SearchResult from '../components/SearchResult';
import SettingsPanel from '../components/SettingsPanel';

/** A single trace entry for the processing pipeline display. */
type TraceEntry = {
  id: number;
  node: string;
  status: 'running' | 'completed' | 'skipped';
  startedAt: number;
  detail?: string;
  hitsCount?: number;
  keywords?: string[];
  variants?: string[];
};

export default function HomePage() {
  const qc = useQueryClient();
  const [query, setQuery] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [streamingTokens, setStreamingTokens] = useState<string | null>(null);
  const [traceEntries, setTraceEntries] = useState<TraceEntry[]>([]);
  const traceIdRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const { data: statusData, isError: statusError } = useQuery<StatusData, Error>({
    queryKey: ['status'],
    queryFn: status,
    retry: false,
    refetchInterval: 2_000,
  });

  const { data: llmSettings } = useQuery<LlmSettings, Error>({
    queryKey: ['llm-settings'],
    queryFn: fetchLlmSettings,
    retry: false,
  });

  const settingsMutation = useMutation({
    mutationFn: updateLlmSettings,
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['llm-settings'] });
      await qc.invalidateQueries({ queryKey: ['llm-models'] });
      setSettingsOpen(false);
    },
  });

  const isTokenEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'token' }> =>
    event.event === 'token';
  const isNodeStartEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'node_start' }> =>
    event.event === 'node_start';
  const isNodeResultEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'node_result' }> =>
    event.event === 'node_result';
  const isDoneEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'done' }> =>
    event.event === 'done';
  const isErrorEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'error' }> =>
    event.event === 'error';

  /** Add a trace entry for a node that just started. */
  const addTraceStart = useCallback((data: NodeStartData) => {
    const id = ++traceIdRef.current;
    const entry: TraceEntry = {
      id,
      node: data.node,
      status: 'running',
      startedAt: Date.now(),
      variants: data.variants,
    };
    setTraceEntries((prev) => [...prev, entry]);
  }, []);

  /** Update the trace entry for a node that completed. */
  const updateTraceResult = useCallback((data: NodeResultData) => {
    setTraceEntries((prev) => {
      // Find the last running entry for this node
      const idx = [...prev].reverse().findIndex((e) => e.node === data.node && e.status === 'running');
      if (idx === -1) return prev;
      const realIdx = prev.length - 1 - idx;
      const updated = [...prev];
      updated[realIdx] = {
        ...updated[realIdx],
        status: data.note === 'no vector hits' || data.note === 'no lexical hits' || data.note === 'no new keywords' ? 'skipped' : 'completed',
        hitsCount: data.hits_count ?? data.fused_count,
        keywords: data.keywords,
        detail: data.note,
      };
      return updated;
    });
  }, []);

  const runSearch = useCallback(
    async ({ query }: { query: string }) => {
      // Cancel previous stream if still active
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;
      setStreamingTokens(null);
      setStreamingTokens(null);
      setTraceEntries([]);
      traceIdRef.current = 0;

      const streamedData: SearchData = {
        answer: null,
        chunks: [],
      };

      try {
        for await (const event of streamSearch(query, 'ai_answer', undefined, controller.signal)) {
          if (controller.signal.aborted) break;

          if (isTokenEvent(event)) {
            setStreamingTokens((current) => (current ?? '') + event.data.token);
          } else if (isNodeStartEvent(event)) {
            addTraceStart(event.data);
          } else if (isNodeResultEvent(event)) {
            updateTraceResult(event.data);
          } else if (isDoneEvent(event)) {
            setStreamingTokens(null);
            return event.data;
          } else if (isErrorEvent(event)) {
            throw new Error(event.data.message);
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return streamedData;
        }
        throw err;
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }

      return streamedData;
    },
    [addTraceStart, updateTraceResult]
  );

  const searchMutation = useMutation({
    mutationFn: runSearch,
  });

  const ingestMutation = useMutation({
    mutationFn: () => ingest(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['status'] }),
  });

  const onSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!query.trim()) return;
      searchMutation.mutate({ query: query.trim() });
    },
    [query, searchMutation]
  );

  const backendOk = !statusError;
  const isIndexing = ingestMutation.isPending || statusData?.is_indexing === true;

  /** Human-readable node labels for the trace panel. */
  const nodeLabels: Record<string, string> = {
    normalize_query: 'Normalize query',
    vector_search: 'Vector search (semantic)',
    extract_keywords_from_vector: 'Extract keywords from vector hits',
    lexical_search_pass1: 'Lexical search pass 1 (keyword)',
    extract_keywords_from_lexical: 'Extract keywords from lexical hits',
    lexical_search_pass2: 'Lexical search pass 2 (keyword)',
    fuse_results: 'Fuse results',
    generate_answer: 'Generate answer (LLM)',
  };

  return (
    <div className="container">
      <header className="app-header">
        <h1>
████████╗ ███████╗  ██████╗ ██╗  ██╗   ██╗  ██╗ ███████╗ ██╗      ██████╗    ███████╗ ███████╗  █████╗  ██████╗   ██████╗  ██╗  ██╗<br/>
╚══██╔══╝ ██╔════╝ ██╔════╝ ██║  ██║   ██║  ██║ ██╔════╝ ██║      ██╔══██╗   ██╔════╝ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔════╝  ██║  ██║<br/>
&nbsp;&nbsp;&nbsp;██║    █████╗   ██║      ███████║   ███████║ █████╗   ██║      ██████╔╝   ███████║ █████╗   ███████║ ██████╔╝ ██║       ███████║<br/>
&nbsp;&nbsp;&nbsp;██║    ██╔══╝   ██║      ██╔══██║   ██╔══██║ ██╔══╝   ██║      ██╔═══╝    ╚════██║ ██╔══╝   ██╔══██║ ██╔══██╗ ██║       ██╔══██║<br/>
&nbsp;&nbsp;&nbsp;██║    ███████╗ ╚██████╗ ██║  ██║   ██║  ██║ ███████╗ ███████╗ ██║        ███████║ ███████╗ ██║  ██║ ██║  ██║ ╚██████╗  ██║  ██║<br/>
&nbsp;&nbsp;&nbsp;╚═╝    ╚══════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚═╝        ╚══════╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝  ╚═╝  ╚═╝
 </h1>
        <div className="status">
          <span className="badge">Backend: {backendOk ? 'connected' : 'offline'}</span>
          {backendOk && (
            <>
              <span className="badge">Indexed files: {statusData?.indexed_files ?? 0}</span>
              <span className="badge">Chunks: {statusData?.indexed_chunks ?? 0}</span>
              {isIndexing && <span className="badge">Indexing in progress</span>}
            </>
          )}
          <button
            className="secondary"
            onClick={() => ingestMutation.mutate()}
            disabled={isIndexing || !backendOk}
          >
            {isIndexing ? 'Indexing...' : 'Run ingestion'}
          </button>
          <button
            className="settings-trigger secondary"
            type="button"
            aria-label="Open model settings"
            title="Model settings"
            onClick={() => setSettingsOpen(true)}
            disabled={!backendOk || !llmSettings}
          >
            ⚙
          </button>
        </div>
      </header>

      {ingestMutation.error && (
        <p className="empty">{ingestMutation.error.message}</p>
      )}

      {settingsOpen && llmSettings && (
        <SettingsPanel
          settings={llmSettings}
          isSaving={settingsMutation.isPending}
          saveError={settingsMutation.error}
          onClose={() => {
            setSettingsOpen(false);
            settingsMutation.reset();
          }}
          onSave={(settings) => settingsMutation.mutate(settings)}
        />
      )}

      <form className="search-bar" onSubmit={onSubmit}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about documentation..."
          disabled={searchMutation.isPending || !backendOk}
        />
        <button type="submit" disabled={searchMutation.isPending || !backendOk}>
          {searchMutation.isPending ? 'Searching...' : 'Search'}
        </button>
      </form>

      {streamingTokens && (
        <section className="result">
          <article className="answer streaming">
            <h3>Answer</h3>
            <p className="streaming-content streaming-active">{streamingTokens}</p>
          </article>
        </section>
      )}

      {searchMutation.data && (
        <section className="result">
          <SearchResult data={searchMutation.data} />
        </section>
      )}

      {searchMutation.error && (
        <p className="empty">{(searchMutation.error as Error).message}</p>
      )}

      {/* Processing pipeline trace — shows all graph nodes in real-time */}
      {traceEntries.length > 0 && (
        <section className="result">
          <details className="trace-panel">
            <summary className="trace-header">
              <h3>Processing pipeline</h3>
              <span className="trace-count">{traceEntries.length} steps</span>
            </summary>
            <ul className="trace-list">
              {traceEntries.map((entry) => (
                <li
                  key={entry.id}
                  className={`trace-item trace-${entry.status}`}
                >
                  <div className="trace-main">
                    <strong>{nodeLabels[entry.node] || entry.node}</strong>
                    <span className="trace-status-badge">{entry.status}</span>
                  </div>
                  <div className="trace-meta">
                    {entry.hitsCount !== undefined && (
                      <span className="trace-stat">{entry.hitsCount} hits</span>
                    )}
                    {entry.keywords && entry.keywords.length > 0 && (
                      <span className="trace-stat">{entry.keywords.length} keywords</span>
                    )}
                    {entry.variants && entry.variants.length > 0 && (
                      <span className="trace-stat">{entry.variants.length} variants</span>
                    )}
                    {entry.detail && (
                      <span className="trace-stat trace-note">{entry.detail}</span>
                    )}
                  </div>
                  {/* Show keywords/variants as expandable details */}
                  {(entry.keywords && entry.keywords.length > 0) && (
                    <details className="trace-details">
                      <summary>Keywords</summary>
                      <pre>{entry.keywords.join(', ')}</pre>
                    </details>
                  )}
                  {(entry.variants && entry.variants.length > 0) && (
                    <details className="trace-details">
                      <summary>Query variants</summary>
                      <pre>{entry.variants.join('\n')}</pre>
                    </details>
                  )}
                </li>
              ))}
            </ul>
          </details>
        </section>
      )}
    </div>
  );
}
