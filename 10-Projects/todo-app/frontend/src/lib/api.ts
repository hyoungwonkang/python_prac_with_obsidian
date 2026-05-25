const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Todo = {
  id: number;
  title: string;
  description: string | null;
  done: boolean;
  created_at: string;
};

export async function getTodos(): Promise<Todo[]> {
  const res = await fetch(`${API_BASE_URL}/api/todos`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch todos: ${res.status}`);
  }
  return res.json();
}

export async function createTodo(title: string): Promise<Todo> {
  const res = await fetch(`${API_BASE_URL}/api/todos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) {
    throw new Error(`Failed to create todo: ${res.status}`);
  }
  return res.json();
}

export async function updateTodo(
  id: number,
  patch: Partial<Pick<Todo, "title" | "description" | "done">>,
): Promise<Todo> {
  const res = await fetch(`${API_BASE_URL}/api/todos/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    throw new Error(`Failed to update todo: ${res.status}`);
  }
  return res.json();
}

export async function deleteTodo(id: number): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/todos/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Failed to delete todo: ${res.status}`);
  }
}
