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
