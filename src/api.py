from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from process import process_floorplan

import tempfile
import uuid
import os

app = FastAPI()


@app.post("/process")
async def process_image(
    file: UploadFile = File(...)
):
    temp_input = f"/tmp/{uuid.uuid4()}.png"
    temp_output = f"/tmp/{uuid.uuid4()}_out.png"

    content = await file.read()

    with open(temp_input, "wb") as f:
        f.write(content)

    process_floorplan(
        temp_input,
        temp_output
    )

    return FileResponse(
        temp_output,
        media_type="image/png"
    )