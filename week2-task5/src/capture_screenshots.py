from playwright.sync_api import sync_playwright
import time
import os

def run(playwright):
    os.makedirs('outputs', exist_ok=True)
    browser = playwright.chromium.launch()
    page = browser.new_page()
    # Set viewport so swagger UI fits nicely
    page.set_viewport_size({"width": 1280, "height": 1080})
    page.goto('http://127.0.0.1:8000/docs')
    time.sleep(2) # Wait for Swagger UI to render
    
    # --- POST /match ---
    page.click('#operations-default-match_candidate_match_post .opblock-summary')
    time.sleep(1)
    page.click('#operations-default-match_candidate_match_post .try-out__btn')
    time.sleep(1)
    # Clear the textarea and fill
    textarea = page.locator('#operations-default-match_candidate_match_post textarea.body-param__text')
    textarea.fill('{\n  "job_id": "J001",\n  "student_id": "S016"\n}')
    time.sleep(1)
    page.click('#operations-default-match_candidate_match_post .execute')
    time.sleep(2) # wait for response
    # Take screenshot
    op_block = page.locator('#operations-default-match_candidate_match_post')
    op_block.screenshot(path='outputs/api_match_demo.png')
    
    # Close accordion to keep it clean
    page.click('#operations-default-match_candidate_match_post .opblock-summary')
    time.sleep(1)

    # --- POST /rank-candidates ---
    page.click('#operations-default-rank_candidates_rank_candidates_post .opblock-summary')
    time.sleep(1)
    page.click('#operations-default-rank_candidates_rank_candidates_post .try-out__btn')
    time.sleep(1)
    textarea2 = page.locator('#operations-default-rank_candidates_rank_candidates_post textarea.body-param__text')
    textarea2.fill('{\n  "job_id": "J001"\n}')
    time.sleep(1)
    page.click('#operations-default-rank_candidates_rank_candidates_post .execute')
    time.sleep(2)
    op_block2 = page.locator('#operations-default-rank_candidates_rank_candidates_post')
    op_block2.screenshot(path='outputs/api_rank_demo.png')

    browser.close()

if __name__ == '__main__':
    with sync_playwright() as playwright:
        run(playwright)
    print("Screenshots captured to outputs/")
