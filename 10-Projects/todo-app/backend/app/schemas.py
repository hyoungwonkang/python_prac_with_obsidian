from datetime import datetime

from pydantic import BaseModel


class TodoIn(BaseModel):
    title: str
    description: str | None = None


class TodoUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    done: bool | None = None


class TodoOut(BaseModel):
    id: int
    title: str
    description: str | None = None
    done: bool
    created_at: datetime

    model_config = {"from_attributes": True}
