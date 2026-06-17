const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api';

const FETCH_TIMEOUT = 300000; // 5 minutes

export type TraceEvent = {
  type: string;
  step: string;
  status: string;
  message: string;
  details?: Record<string, unknown>;
  duration_ms?: number;
  created_at?: string;
};

export type ReasoningSummary = {
  reasoning_summary?: string[];
  source_notes?: Array<Record<string, unknown>>;
  limitations?: string[];
};

export type SearchData = {
  query?: string;
  mode?: string;
  model?: string;
  answer: string | null;
  chunks: Array<{
    text?: string;
    metadata?: Record<string, unknown>;
    score?: number;
  }>;
  trace?: TraceEvent[];
  reasoning_summary?: string[];
  source_notes?: Array<Record<string, unknown>>;
  limitations?: string[];
};

export type StreamEvent =
  | { event: 'trace'; data: TraceEvent }
  | { event: 'token'; data: { token: string } }
  | { event: 'reasoning_token'; data: { token: string } }
  | { event: 'answer'; data: { answer: string } }
  | { event: 'reasoning'; data: ReasoningSummary }
  | { event: 'done'; data: SearchData }
  | { event: 'error'; data: { message: string } };

export type LlmModel = {
  id: string;
  name: string;
};

export async function fetchModels(): Promise<LlmModel[]> {
  const res = await fetch(`${API_BASE}/llm/models`);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Failed to fetch models: ${res.status} ${text}`);
  }
  return res.json();
}

export async function search(query: string, mode = 'ai_answer', model?: string) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

  try {
    const res = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, model }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`Search failed: ${res.status} ${text}`);
    }
    return (await res.json()) as SearchData;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function* streamSearch(
  query: string,
  mode = 'ai_answer',
  model?: string,
  externalSignal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

  // Если внешний сигнал пришёл — отменяем и внутренний контроллер
  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort(), { once: true });
  }

  try {
    const res = await fetch(`${API_BASE}/search/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, model }),
      signal: controller.signal,
      credentials: 'include',
    });
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`Search stream failed: ${res.status} ${text}`);
    }
    if (!res.body) {
      throw new Error('Search stream response body is empty');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let eventEnd = buffer.indexOf('\n\n');
      while (eventEnd !== -1) {
        const rawEvent = buffer.slice(0, eventEnd);
        buffer = buffer.slice(eventEnd + 2);
        if (rawEvent.trim()) {
          const parsed = parseSseBlock(rawEvent);
          if (parsed) {
            yield parsed;
          }
        }
        eventEnd = buffer.indexOf('\n\n');
      }
    }

    if (buffer.trim()) {
      const parsed = parseSseBlock(buffer);
      if (parsed) {
        yield parsed;
      }
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

function parseSseBlock(rawEvent: string): StreamEvent | null {
  let event = 'message';
  const dataLines: string[] = [];

  for (const line of rawEvent.split(/\r?\n/)) {
    if (!line) continue;
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  try {
    const data = JSON.parse(dataLines.join('\n')) as StreamEvent['data'];
    return { event, data } as StreamEvent;
  } catch (e) {
    console.error('[SSE] JSON parse error:', e, 'raw:', rawEvent);
    return null;
  }
}

export async function ingest() {
  const res = await fetch(`${API_BASE}/ingest`, { method: 'POST' });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Ingest failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function status() {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Status failed: ${res.status} ${text}`);
  }
  return res.json();
}
