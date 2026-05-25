import { getTodos } from "@/lib/api";

export default async function Home() {
  const todos = await getTodos();

  return (
    <main className="max-w-2xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-semibold mb-8 text-zinc-900 dark:text-zinc-50">
        Todo App
      </h1>
      {todos.length === 0 ? (
        <p className="text-zinc-500 dark:text-zinc-400">
          아직 할 일이 없습니다.
        </p>
      ) : (
        <ul className="space-y-2">
          {todos.map((todo) => (
            <li
              key={todo.id}
              className="flex items-center gap-3 p-3 border border-zinc-200 dark:border-zinc-800 rounded-lg bg-white dark:bg-zinc-950"
            >
              <span
                className={
                  todo.done
                    ? "text-zinc-400 line-through dark:text-zinc-600"
                    : "text-zinc-900 dark:text-zinc-100"
                }
              >
                {todo.title}
              </span>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
