from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from . import __version__
from .api.routes import router
from .container import build_container


def create_app() -> FastAPI:
    app = FastAPI(
        title="Millie Recommendation API",
        version=__version__,
        description="Personalized carousel recommendation API",
    )
    app.state.container = build_container()
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("recommendation_service.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()

