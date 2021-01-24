from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id):
    if item_id == 'alexnet':
        return {"model_name": item_id, "message": "Deep Learning FTW!"}
    elif item_id == 'resnet':
        return {"model_name": item_id, "message": "LeCNN all the images!"}