import asyncio
import sys
import os
import json
import requests
from playwright.async_api import async_playwright

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def render_text_to_png(text, title, output_path):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ background-color: #1a1a2e; color: #e0e0e0; font-family: 'Cascadia Code', Consolas, monospace; padding: 30px; }}
            h2 {{ color: #00d2ff; margin-bottom: 4px; }}
            h4 {{ color: #7b68ee; margin-top: 0; }}
            pre {{ white-space: pre-wrap; font-size: 13px; background: #16213e; border-radius: 8px; padding: 16px; border: 1px solid #0f3460; }}
            .req {{ color: #ffd700; }}
            .res {{ color: #50fa7b; }}
            hr {{ border: 1px solid #0f3460; }}
        </style>
    </head>
    <body>
        <h2>PlaceMux Quality Baseline API</h2>
        <h4>{title}</h4>
        <pre>{text}</pre>
    </body>
    </html>
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_html = os.path.join(os.path.dirname(output_path), "_temp.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file:///{os.path.abspath(temp_html).replace(chr(92), '/')}")
        await page.set_viewport_size({"width": 850, "height": 900})
        await page.screenshot(path=output_path, full_page=True)
        await browser.close()
    os.remove(temp_html)

async def main():
    os.makedirs("Outputs", exist_ok=True)
    base = "http://localhost:8001"
    
    # 1. POST /match - successful match
    req1 = {"candidate_id": 10001, "job_id": 1005}
    res1 = requests.post(f"{base}/match", json=req1)
    body1 = json.dumps(res1.json(), indent=2)
    text1 = f'<span class="req">REQUEST</span>\nPOST /match\nContent-Type: application/json\n\n{json.dumps(req1, indent=2)}\n\n<hr/>\n<span class="res">RESPONSE  HTTP {res1.status_code}</span>\n\n{body1}'
    await render_text_to_png(text1, "POST /match — Successful Match", "Outputs/api_match_demo.png")
    print(f"Saved api_match_demo.png  |  response:\n{body1}\n")

    # 2. POST /match - error cases
    req2 = {"candidate_id": 999999, "job_id": 1000}
    res2 = requests.post(f"{base}/match", json=req2)
    body2 = json.dumps(res2.json(), indent=2)
    
    req3 = {"candidate_id": 10005, "job_id": 1002}
    res3 = requests.post(f"{base}/match", json=req3)
    body3 = json.dumps(res3.json(), indent=2)
    
    text2  = f'<span class="req">REQUEST 1 — Unknown Candidate</span>\nPOST /match\n{json.dumps(req2)}\n\n<span class="res">RESPONSE  HTTP {res2.status_code}</span>\n{body2}\n\n<hr/>\n'
    text2 += f'<span class="req">REQUEST 2 — Another Candidate-Job Pair</span>\nPOST /match\n{json.dumps(req3)}\n\n<span class="res">RESPONSE  HTTP {res3.status_code}</span>\n{body3}'
    await render_text_to_png(text2, "POST /match — Multiple Scenarios", "Outputs/api_multi_demo.png")
    print(f"Saved api_multi_demo.png")

if __name__ == '__main__':
    asyncio.run(main())
