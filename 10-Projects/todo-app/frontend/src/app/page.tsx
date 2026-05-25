import { getTodos } from "@/lib/api";
import { AddTodoForm } from "@/components/AddTodoForm";
import { TodoItem } from "@/components/TodoItem";

export default async function Home() {
  const todos = await getTodos();

  return (
    <main className="max-w-2xl mx-auto px-6 py-12">
      <h1 className="text-3xl font-semibold mb-8 text-zinc-900 dark:text-zinc-50">
        Todo App
      </h1>
      <AddTodoForm />
      {todos.length === 0 ? (
        <p className="text-zinc-500 dark:text-zinc-400">
          아직 할 일이 없습니다.
        </p>
      ) : (
        <ul className="space-y-2">
          {todos.map((todo) => (
            <TodoItem key={todo.id} todo={todo} />
          ))}
        </ul>
      )}
    </main>
  );
}
