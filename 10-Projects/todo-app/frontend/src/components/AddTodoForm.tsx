"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { createTodo } from "@/lib/api";

export function AddTodoForm() {
  const [title, setTitle] = useState("");
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = title.trim();
    if (!trimmed) return;
    await createTodo(trimmed);
    setTitle("");
    startTransition(() => router.refresh());
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="할 일을 입력하세요"
        disabled={isPending}
        className="flex-1 px-3 py-2 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={isPending || !title.trim()}
        className="px-4 py-2 bg-zinc-900 dark:bg-zinc-50 text-white dark:text-zinc-900 rounded-lg disabled:opacity-50"
      >
        추가
      </button>
    </form>
  );
}
