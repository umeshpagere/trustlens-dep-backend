import asyncio
from app.services.video.video_pipeline import process_video_text

async def main():
    print("Testing Instagram URL...")
    result = await process_video_text("https://www.instagram.com/p/DVgR1bgEsUx/")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
