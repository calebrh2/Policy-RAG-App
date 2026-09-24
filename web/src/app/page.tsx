"use client";

import { useState } from "react";
import {
  AgentChat,
  type AgentMessage,
  type ChatStatus,
} from "@/components/ui/agent-chat";

type Citation = {
  document_name: string;
  section: string;
  source_pages: string;
};

type AskResponse = {
  answer?: string;
  error?: string;
  citations?: Citation[];
};

export default function HomePage() {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>("ready");
  const [error, setError] = useState<{ message: string; title?: string }>();

  async function onSend(message: { role: "user"; content: string }) {
    setError(undefined);
    setStatus("submitted");
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        parts: [{ type: "text", text: message.content }],
      },
    ]);
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: message.content }),
      });
      const payload = (await response.json()) as AskResponse;
      if (!response.ok || payload.error) {
        setError({
          title: "Request failed",
          message: payload.error || "The policy service did not answer.",
        });
        return;
      }
      const citations = (payload.citations ?? [])
        .map(
          (citation) =>
            `${citation.document_name} — ${citation.section} (pages ${citation.source_pages})`,
        )
        .join("\n");
      const answer = payload.answer ?? "";
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          parts: [
            {
              type: "text",
              text: citations ? `${answer}\n\n${citations}` : answer,
            },
          ],
        },
      ]);
    } catch (caught) {
      setError({
        message:
          caught instanceof Error
            ? caught.message
            : "The policy service did not answer.",
      });
    } finally {
      setStatus("ready");
    }
  }

  return (
    <main className="h-screen bg-neutral-50 p-4 dark:bg-neutral-950">
      <div className="mx-auto flex h-full max-w-[680px] flex-col overflow-hidden rounded-xl border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
        <AgentChat
          messages={messages}
          status={status}
          error={error}
          emptyStatePosition="center"
          onSend={onSend}
          onStop={() => setStatus("ready")}
        />
      </div>
    </main>
  );
}
