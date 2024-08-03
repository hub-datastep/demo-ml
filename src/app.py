import uvicorn
from fastapi import FastAPI

from brands_ner.controller import brands_ner_router
from brands_ner.model import load_ner_model

app = FastAPI()

app.include_router(brands_ner_router, tags=["Brands NER"], prefix="/brands_ner")


@app.on_event("startup")
async def startup_event():
    await load_ner_model()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
