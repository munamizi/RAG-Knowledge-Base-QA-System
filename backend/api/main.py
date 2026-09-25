from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, RedirectResponse

from backend.api.routers import import_router, query_router

PAGE_DIR = Path(__file__).resolve().parents[2] / "frontend"



# 创建FastAPI实例
app = FastAPI(
    title="RAG知识库",
    description="知识库导入与检索问答服务",
)

# 路由分组
app.include_router(query_router)
app.include_router(import_router)


# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse("/chat.html")


@app.get("/chat.html", include_in_schema=False)
async def chat_page():
    return html_response("chat.html")


@app.get("/import.html", include_in_schema=False)
async def import_page():
    return html_response("import.html")



def html_response(filename: str) -> FileResponse:
    html_path = PAGE_DIR / filename

    if not html_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"没有找到页面：{html_path}",
        )

    return FileResponse(html_path)



if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)
