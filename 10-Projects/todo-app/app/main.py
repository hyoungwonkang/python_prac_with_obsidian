from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import select

from app.database import async_session, engine
from app.models import Base, Todo
from app.schemas import TodoIn, TodoOut, TodoUpdate


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Todo App", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/todos", response_model=TodoOut, status_code=201)
async def create_todo(data: TodoIn):
    async with async_session() as session:
        todo = Todo(title=data.title, description=data.description)
        session.add(todo)
        await session.commit()
        await session.refresh(todo)
        return todo


@app.get("/api/todos", response_model=list[TodoOut])
async def list_todos():
    async with async_session() as session:
        result = await session.execute(select(Todo).order_by(Todo.created_at.desc()))
        return result.scalars().all()


@app.get("/api/todos/{todo_id}", response_model=TodoOut)
async def get_todo(todo_id: int):
    async with async_session() as session:
        todo = await session.get(Todo, todo_id)
        if not todo:
            raise HTTPException(status_code=404, detail="Todo not found")
        return todo


@app.patch("/api/todos/{todo_id}", response_model=TodoOut)
async def update_todo(todo_id: int, data: TodoUpdate):
    async with async_session() as session:
        todo = await session.get(Todo, todo_id)
        if not todo:
            raise HTTPException(status_code=404, detail="Todo not found")
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(todo, key, value)
        await session.commit()
        await session.refresh(todo)
        return todo


@app.delete("/api/todos/{todo_id}", status_code=204)
async def delete_todo(todo_id: int):
    async with async_session() as session:
        todo = await session.get(Todo, todo_id)
        if not todo:
            raise HTTPException(status_code=404, detail="Todo not found")
        await session.delete(todo)
        await session.commit()
