"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
    ImageIcon,
    FileUp,
    Figma,
    MonitorIcon,
    CircleUserRound,
    ArrowUpIcon,
    LoaderCircle,
    Paperclip,
    PlusIcon,
} from "lucide-react";

interface UseAutoResizeTextareaProps {
    minHeight: number;
    maxHeight?: number;
}

function useAutoResizeTextarea({
    minHeight,
    maxHeight,
}: UseAutoResizeTextareaProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const adjustHeight = useCallback(
        (reset?: boolean) => {
            const textarea = textareaRef.current;
            if (!textarea) return;

            if (reset) {
                textarea.style.height = `${minHeight}px`;
                return;
            }

            // Temporarily shrink to get the right scrollHeight
            textarea.style.height = `${minHeight}px`;

            // Calculate new height
            const newHeight = Math.max(
                minHeight,
                Math.min(
                    textarea.scrollHeight,
                    maxHeight ?? Number.POSITIVE_INFINITY
                )
            );

            textarea.style.height = `${newHeight}px`;
        },
        [minHeight, maxHeight]
    );

    useEffect(() => {
        // Set initial height
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = `${minHeight}px`;
        }
    }, [minHeight]);

    // Adjust height on window resize
    useEffect(() => {
        const handleResize = () => adjustHeight();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, [adjustHeight]);

    return { textareaRef, adjustHeight };
}

interface Citation {
    chunk_id: string;
    document_title: string;
    section_path: string;
}

interface QueryAnswer {
    text: string;
    supported: boolean;
    citations: Citation[];
}

interface ChatTurn {
    id: number;
    query: string;
    answer: QueryAnswer | null;
    error: string | null;
    pending: boolean;
}

export function VercelV0Chat() {
    const [value, setValue] = useState("");
    const [loading, setLoading] = useState(false);
    const [turns, setTurns] = useState<ChatTurn[]>([]);
    const nextTurnId = useRef(1);
    const historyRef = useRef<HTMLDivElement>(null);
    const { textareaRef, adjustHeight } = useAutoResizeTextarea({
        minHeight: 60,
        maxHeight: 200,
    });

    useEffect(() => {
        const history = historyRef.current;
        if (!history) return;
        history.scrollTop = history.scrollHeight;
    }, [turns]);

    const submit = async () => {
        const text = value.trim();
        if (!text || loading) return;

        const id = nextTurnId.current++;
        setTurns((current) => [
            ...current,
            { id, query: text, answer: null, error: null, pending: true },
        ]);
        setValue("");
        adjustHeight(true);
        setLoading(true);

        const finish = (update: Partial<ChatTurn>) => {
            setTurns((current) =>
                current.map((turn) =>
                    turn.id === id ? { ...turn, pending: false, ...update } : turn
                )
            );
        };

        try {
            const response = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                finish({ error: "The query failed" });
                return;
            }

            const data = (await response.json()) as QueryAnswer;
            finish({ answer: data });
        } catch {
            finish({ error: "Could not reach the API" });
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submit();
        }
    };

    return (
        <div className="flex flex-col items-center w-full max-w-4xl mx-auto p-4 space-y-8">
            <h1 className="text-4xl font-bold text-black dark:text-white">
                Ask the policy corpus
            </h1>

            <div className="w-full">
                <div className="relative flex flex-col bg-neutral-900 rounded-xl border border-neutral-800">
                    {turns.length > 0 ? (
                        <div
                            ref={historyRef}
                            className="max-h-96 space-y-4 overflow-y-auto px-4 pb-3 pt-4"
                        >
                            {turns.map((turn) => (
                                <div key={turn.id} className="space-y-3">
                                    <div className="flex justify-end">
                                        <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-neutral-800 px-3 py-2 text-sm text-white">
                                            {turn.query}
                                        </p>
                                    </div>
                                    {turn.pending ? (
                                        <div className="flex items-center text-neutral-400">
                                            <LoaderCircle className="h-4 w-4 animate-spin" />
                                            <span className="sr-only">
                                                Searching the policies
                                            </span>
                                        </div>
                                    ) : null}
                                    {turn.error ? (
                                        <p className="text-sm text-neutral-400">
                                            {turn.error}
                                        </p>
                                    ) : null}
                                    {turn.answer ? (
                                        <div className="space-y-3 text-left text-sm text-neutral-200">
                                            <p className="whitespace-pre-wrap">
                                                {turn.answer.text}
                                            </p>
                                            {turn.answer.citations.length > 0 ? (
                                                <div className="space-y-2">
                                                    <p className="text-xs uppercase tracking-wide text-neutral-500">
                                                        Sources
                                                    </p>
                                                    <ul className="space-y-2">
                                                        {turn.answer.citations.map(
                                                            (citation, index) => (
                                                                <li
                                                                    key={`${citation.chunk_id}-${index}`}
                                                                    className="rounded-lg border border-neutral-800 px-3 py-2"
                                                                >
                                                                    <p className="text-neutral-100">
                                                                        {
                                                                            citation.document_title
                                                                        }
                                                                        {citation.section_path
                                                                            ? ` — ${citation.section_path}`
                                                                            : ""}
                                                                    </p>
                                                                </li>
                                                            )
                                                        )}
                                                    </ul>
                                                </div>
                                            ) : null}
                                        </div>
                                    ) : null}
                                </div>
                            ))}
                        </div>
                    ) : null}

                    <div
                        className={cn(
                            turns.length > 0 && "border-t border-neutral-800"
                        )}
                    >
                    <div className="overflow-y-auto">
                        <Textarea
                            ref={textareaRef}
                            value={value}
                            onChange={(e) => {
                                setValue(e.target.value);
                                adjustHeight();
                            }}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask a question about the policies..."
                            className={cn(
                                "w-full px-4 py-3",
                                "resize-none",
                                "bg-transparent",
                                "border-none",
                                "text-white text-sm",
                                "focus:outline-none",
                                "focus-visible:ring-0 focus-visible:ring-offset-0",
                                "placeholder:text-neutral-500 placeholder:text-sm",
                                "min-h-[60px]"
                            )}
                            style={{
                                overflow: "hidden",
                            }}
                        />
                    </div>

                    <div className="flex items-center justify-between p-3">
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                className="group p-2 hover:bg-neutral-800 rounded-lg transition-colors flex items-center gap-1"
                            >
                                <Paperclip className="w-4 h-4 text-white" />
                                <span className="text-xs text-zinc-400 hidden group-hover:inline transition-opacity">
                                    Attach
                                </span>
                            </button>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                className="px-2 py-1 rounded-lg text-sm text-zinc-400 transition-colors border border-dashed border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800 flex items-center justify-between gap-1"
                            >
                                <PlusIcon className="w-4 h-4" />
                                Project
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    void submit();
                                }}
                                className={cn(
                                    "px-1.5 py-1.5 rounded-lg text-sm transition-colors border border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800 flex items-center justify-between gap-1",
                                    value.trim()
                                        ? "bg-white text-black"
                                        : "text-zinc-400"
                                )}
                            >
                                <ArrowUpIcon
                                    className={cn(
                                        "w-4 h-4",
                                        value.trim()
                                            ? "text-black"
                                            : "text-zinc-400"
                                    )}
                                />
                                <span className="sr-only">Send</span>
                            </button>
                        </div>
                    </div>
                    </div>
                </div>

                <div className="flex flex-wrap items-center justify-center gap-3 mt-4">
                    <ActionButton
                        icon={<ImageIcon className="w-4 h-4" />}
                        label="Clone a Screenshot"
                    />
                    <ActionButton
                        icon={<Figma className="w-4 h-4" />}
                        label="Import from Figma"
                    />
                    <ActionButton
                        icon={<FileUp className="w-4 h-4" />}
                        label="Upload a Project"
                    />
                    <ActionButton
                        icon={<MonitorIcon className="w-4 h-4" />}
                        label="Landing Page"
                    />
                    <ActionButton
                        icon={<CircleUserRound className="w-4 h-4" />}
                        label="Sign Up Form"
                    />
                </div>
            </div>
        </div>
    );
}

interface ActionButtonProps {
    icon: ReactNode;
    label: string;
}

function ActionButton({ icon, label }: ActionButtonProps) {
    return (
        <button
            type="button"
            className="flex items-center gap-2 px-4 py-2 bg-neutral-900 hover:bg-neutral-800 rounded-full border border-neutral-800 text-neutral-400 hover:text-white transition-colors"
        >
            {icon}
            <span className="text-xs">{label}</span>
        </button>
    );
}
