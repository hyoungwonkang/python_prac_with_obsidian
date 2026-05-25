"use client";

import { useTransition } from "react";
import { useRouter } from "next/navigation";
import { updateTodo, deleteTodo, type Todo } from "@/lib/api";

export function TodoItem({ todo }: { todo: Todo }) {
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  async function handleToggle() {
    await updateTodo(todo.id, { done: !todo.done });
    startTransition(() => router.refresh());
  }

  async function handleDelete() {
    await deleteTodo(todo.id);
    startTransition(() => router.refresh());
  }

  return (
    <li className="flex items-center gap-3 p-3 border border-zinc-200 dark:border-zinc-800 rounded-lg bg-white dark:bg-zinc-950">
      <input
        type="checkbox"
        checked={todo.done}
        onChange={handleToggle}
        disabled={isPending}
        className="w-4 h-4 cursor-pointer"
      />
      <span
        className={`flex-1 ${
          todo.done
            ? "text-zinc-400 line-through dark:text-zinc-600"
            : "text-zinc-900 dark:text-zinc-100"
        }`}
      >
        {todo.title}
      </span>
      <button
        onClick={handleDelete}
        disabled={isPending}
        className="text-sm text-red-600 dark:text-red-400 hover:underline disabled:opacity-50"
      >
        삭제
      </button>
    </li>
  );
}
