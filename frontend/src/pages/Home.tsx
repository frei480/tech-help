import { useCallback, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchModels,
  ingest,
  status,
  streamSearch,
  type LlmModel,
  type ReasoningSummary,
  type SearchData,
  type StreamEvent,
} from '../api';
import ProcessTrace from '../components/ProcessTrace';
import SearchResult from '../components/SearchResult';

type StatusData = { indexed_files: number; indexed_chunks: number } | Record<string, never>;

export default function HomePage() {
  const qc = useQueryClient();
  const [query, setQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [streamReasoning, setStreamReasoning] = useState<ReasoningSummary | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState<string | null>(null);
  const [streamReasoningTokens, setStreamReasoningTokens] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const { data: statusData, isError: statusError } = useQuery<StatusData, Error>({
    queryKey: ['status'],
    queryFn: status,
    retry: false,
  });

  const { data: models = [] } = useQuery<LlmModel[], Error>({
    queryKey: ['llm-models'],
    queryFn: fetchModels,
    retry: false,
    staleTime: 60_000, // обновлять не чаще чем раз в минуту
  });

  const isTraceEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'trace' }> => event.event === 'trace';
  const isTokenEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'token' }> => event.event === 'token';
  const isAnswerEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'answer' }> => event.event === 'answer';
  const isReasoningEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'reasoning' }> => event.event === 'reasoning';
  const isDoneEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'done' }> => event.event === 'done';
  const isErrorEvent = (event: StreamEvent): event is Extract<StreamEvent, { event: 'error' }> => event.event === 'error';

  const runSearch = useCallback(
    async ({ query, model }: { query: string; model?: string }) => {
      // Отменяем предыдущий стрим, если он ещё активен
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setStreamReasoning(null);
      setStreamingAnswer(null);
      setStreamReasoningTokens(null);

      const streamedData: SearchData = {
        answer: null,
        chunks: [],
        trace: [],
        reasoning_summary: [],
        source_notes: [],
        limitations: [],
      };

      try {
        for await (const event of streamSearch(query, 'ai_answer', model, controller.signal)) {
          if (controller.signal.aborted) break;

          if (isTraceEvent(event)) {
            streamedData.trace = [...(streamedData.trace ?? []), event.data];
          } else if (event.event === 'reasoning_token') {
            setStreamReasoningTokens((current) => (current ?? '') + event.data.token);
          } else if (isTokenEvent(event)) {
            setStreamingAnswer((current) => (current ?? '') + event.data.token);
          } else if (isAnswerEvent(event)) {
            streamedData.answer = event.data.answer;
            setStreamingAnswer(event.data.answer ?? null);
          } else if (isReasoningEvent(event)) {
            streamedData.reasoning_summary = event.data.reasoning_summary ?? [];
            streamedData.source_notes = event.data.source_notes ?? [];
            streamedData.limitations = event.data.limitations ?? [];
            setStreamReasoning(event.data);
          } else if (isDoneEvent(event)) {
            setStreamingAnswer(null);
            return event.data;
          } else if (isErrorEvent(event)) {
            throw new Error(event.data.message);
          }
        }
      } catch (err) {
        // Если ошибка из-за отмены — не пробрасываем её дальше
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
    []
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
      searchMutation.mutate({ query: query.trim(), model: selectedModel || undefined });
    },
    [query, selectedModel, searchMutation]
  );

  const backendOk = !statusError;

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
              <span className="badge">Indexed files: {(statusData as StatusData | undefined)?.indexed_files ?? 0}</span>
              <span className="badge">Chunks: {(statusData as StatusData | undefined)?.indexed_chunks ?? 0}</span>
            </>
          )}
          <button
            className="secondary"
            onClick={() => ingestMutation.mutate()}
            disabled={ingestMutation.isPending || !backendOk}
          >
            {ingestMutation.isPending ? 'Indexing...' : 'Run ingestion'}
          </button>
        </div>
      </header>

      <form className="search-bar" onSubmit={onSubmit}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about documentation..."
          disabled={searchMutation.isPending || !backendOk}
        />
        <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
          <option value="">Default ({models.length > 0 ? models[0]?.name ?? '...' : '...'})</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
        <button type="submit" disabled={searchMutation.isPending || !backendOk}>
          {searchMutation.isPending ? 'Searching...' : 'Search'}
        </button>
      </form>

      <section className="result">
        <ProcessTrace
          reasoning={streamReasoning}
          isLoading={searchMutation.isPending}
          streamingAnswer={streamingAnswer}
          reasoningTokens={streamReasoningTokens}
        />
      </section>

      {searchMutation.data && (
        <section className="result">
          <SearchResult data={searchMutation.data} />
        </section>
      )}

      {searchMutation.error && (
        <p className="empty">{(searchMutation.error as Error).message}</p>
      )}
    </div>
  );
}
