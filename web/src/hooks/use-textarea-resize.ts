import { useEffect, useRef } from "react";

export function useTextareaResize(value: string, rows = 1) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const lineHeight = Number.parseFloat(getComputedStyle(textarea).lineHeight) || 20;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.max(textarea.scrollHeight, lineHeight * rows)}px`;
  }, [value, rows]);

  return textareaRef;
}
