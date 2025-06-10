async def save_upload_file(upload_file, destination):
    with open(destination, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
