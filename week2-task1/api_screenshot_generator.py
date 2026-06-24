import asyncio
import sys
import os
import requests
import time
import subprocess
import json
from playwright.async_api import async_playwright

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def render_text_to_png(text, title, output_path):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; padding: 20px; }}
            h3 {{ color: #569cd6; }}
            pre {{ white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <h3>{title}</h3>
        <pre>{text}</pre>
    </body>
    </html>
    """
    with open("temp.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file:///{os.path.abspath('temp.html').replace(chr(92), '/')}")
        await page.set_viewport_size({"width": 800, "height": 1000})
        await page.screenshot(path=output_path, full_page=True)
        await browser.close()
    os.remove("temp.html")

async def main():
    print("Starting server...")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--port", "8000"], cwd=os.getcwd())
    time.sleep(3) # Wait for server to start
    
    try:
        # POST /match
        print("Testing POST /match...")
        student_id = 1
        job_id = 1001
        match_req = {"student_id": student_id, "job_id": job_id}
        res = requests.post("http://localhost:8000/match", json=match_req)
        match_out = f"POST /match HTTP/1.1\\nHost: localhost:8000\\nBody: {json.dumps(match_req)}\\n\\n"
        match_out += f"HTTP/1.1 {res.status_code}\\n\\n{json.dumps(res.json(), indent=2)}"
        await render_text_to_png(match_out, "POST /match", "Outputs/api_match_demo.png")
        print("Saved Outputs/api_match_demo.png")

        # GET /top-candidates/1001
        print("Testing GET /top-candidates...")
        res = requests.get("http://localhost:8000/top-candidates/1001?top_n=3")
        rank_out = f"GET /top-candidates/1001?top_n=3 HTTP/1.1\\nHost: localhost:8000\\n\\n"
        rank_out += f"HTTP/1.1 {res.status_code}\\n\\n{json.dumps(res.json(), indent=2)}"
        await render_text_to_png(rank_out, "GET /top-candidates/1001", "Outputs/api_rank_demo.png")
        print("Saved Outputs/api_rank_demo.png")

    finally:
        server.terminate()
        server.wait()

if __name__ == '__main__':
    asyncio.run(main())
