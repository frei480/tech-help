import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchEmbeddingModels, fetchModels, type LlmModel, type LlmSettings } from '../api';

type SettingsPanelProps = {
  settings: LlmSettings;
  isSaving: boolean;
  saveError: Error | null;
  onClose: () => void;
  onSave: (settings: LlmSettings) => void;
};

export default function SettingsPanel({
  settings,
  isSaving,
  saveError,
  onClose,
  onSave,
}: SettingsPanelProps) {
  const [draft, setDraft] = useState(settings);
  const {
    data: models = [],
    isLoading: isLoadingModels,
    error: modelsError,
  } = useQuery<LlmModel[], Error>({
    queryKey: ['llm-models', draft.provider],
    queryFn: () => fetchModels(draft.provider),
    staleTime: 60_000,
  });
  const {
    data: openRouterEmbeddingModels = [],
    isLoading: isLoadingEmbeddingModels,
    error: embeddingModelsError,
  } = useQuery<LlmModel[], Error>({
    queryKey: ['llm-embedding-models', draft.provider],
    queryFn: () => fetchEmbeddingModels(draft.provider),
    enabled: draft.provider === 'openrouter',
    staleTime: 60_000,
  });
  const embeddingModels =
    draft.provider === 'openrouter' ? openRouterEmbeddingModels : models;
  const isLoading =
    isLoadingModels || (draft.provider === 'openrouter' && isLoadingEmbeddingModels);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const modelOptions = (selected: string, options: LlmModel[]) => (
    <>
      {!options.some((model) => model.id === selected) && selected && (
        <option value={selected}>{selected} (current)</option>
      )}
      {options.map((model) => (
        <option key={model.id} value={model.id}>
          {model.name}
        </option>
      ))}
    </>
  );

  return (
    <div
      className="settings-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
      >
        <div className="settings-header">
          <h2 id="settings-title">Model settings</h2>
          <button className="settings-close secondary" type="button" onClick={onClose} aria-label="Close settings">
            ×
          </button>
        </div>

        <form
          className="settings-form"
          onSubmit={(event) => {
            event.preventDefault();
            onSave(draft);
          }}
        >
          <label>
            Provider
            <select
              autoFocus
              value={draft.provider}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  provider: event.target.value as LlmSettings['provider'],
                }))
              }
            >
              <option value="lm_studio">LM Studio</option>
              <option value="openrouter">OpenRouter</option>
            </select>
          </label>

          <label>
            LLM model
            <select
              value={draft.chat_model}
              onChange={(event) =>
                setDraft((current) => ({ ...current, chat_model: event.target.value }))
              }
              disabled={isLoading || models.length === 0}
            >
              {modelOptions(draft.chat_model, models)}
            </select>
          </label>

          <label>
            Embeddings model
            <select
              value={draft.embedding_model}
              onChange={(event) =>
                setDraft((current) => ({ ...current, embedding_model: event.target.value }))
              }
              disabled={isLoading || embeddingModels.length === 0}
            >
              {modelOptions(draft.embedding_model, embeddingModels)}
            </select>
          </label>

          {isLoading && <p className="settings-help">Loading models…</p>}
          {modelsError && <p className="settings-error">{modelsError.message}</p>}
          {draft.provider === 'openrouter' && embeddingModelsError && (
            <p className="settings-error">{embeddingModelsError.message}</p>
          )}
          <p className="settings-help">
            Changing the embeddings model requires re-indexing your documents.
          </p>
          {saveError && <p className="settings-error">{saveError.message}</p>}

          <div className="settings-actions">
            <button type="button" className="secondary" onClick={onClose} disabled={isSaving}>
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSaving || isLoading || models.length === 0 || embeddingModels.length === 0}
            >
              {isSaving ? 'Saving…' : 'Save settings'}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
