from os import getenv

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

load_dotenv()

DATABASE_URL = getenv("DATABASE_URL", "sqlite+aiosqlite:///./todo.db")

connect_args = {}
if DATABASE_URL.startswith("postgresql"):
    connect_args["statement_cache_size"] = 0
    connect_args["prepared_statement_cache_size"] = 0

engine = create_async_engine(DATABASE_URL, connect_args=connect_args)
async_session = async_sessionmaker(engine, expire_on_commit=False)
