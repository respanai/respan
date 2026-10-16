"""Fill a sign-up form: text, password, select, checkbox, submit - mirrors
Playwright for Python's "Actions" page (fill, select_option, check, click).

    python scenarios/03_forms.py
"""

from playwright.sync_api import expect, sync_playwright

from _harness import SITE, Expect, Scenario, launch_options, main

PASSWORD = "correct-horse-battery-staple-42"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_options())
        page = browser.new_page()
        page.goto(f"{SITE}/form.html")
        page.get_by_label("Email").fill("ada@example.com")
        page.get_by_label("Password").fill(PASSWORD)
        page.get_by_label("Plan").select_option("pro")
        page.get_by_label("I accept the terms").check()
        page.get_by_role("button", name="Create account").click()
        # The form submits by GET: the password ends up in the page URL too.
        expect(page).to_have_title("Welcome")
        browser.close()


SCENARIO = Scenario(
    name="forms",
    title="Fill and submit a form",
    source="docs/input",
    look_for="Locator.fill shows the email; the password fill shows <masked>; the URL after submit shows <secret>password</secret>",
    run=run,
    expect=Expect(
        calls_include=(
            "Locator.fill",
            "Locator.select_option",
            "Locator.check",
            "Locator.click",
            "PageAssertions.to_have_title",
        ),
        forbidden=(PASSWORD,),
    ),
)

if __name__ == "__main__":
    main(SCENARIO)
