"""persisted_login - mirrors cookbooks/persisted-login.

observe() for selectors, fill credentials without the model ever seeing them,
then a second session on the same profile (user_data_dir) reads the page state.

    python scenarios/02_persisted_login.py
"""

import shutil
import tempfile

from pydantic import BaseModel

from _harness import Expect, Scenario, fixtures_url, main, with_stagehand

PASSWORD = "hunter2-correct-horse-4471"


class SignedIn(BaseModel):
    signed_in: bool
    message: str


async def run():
    site = fixtures_url()
    profile = tempfile.mkdtemp(prefix="stagehand-login-")
    try:

        async def sign_in(stagehand, page, browser):
            await page.goto(f"{site}/login.html")
            email = (await stagehand.observe("find the email input")).data
            password = (await stagehand.observe("find the password input")).data
            await page.locator(email[0].selector).fill("ada@example.com")
            await page.locator(password[0].selector).fill(PASSWORD)
            await stagehand.act("click the sign in button")

        async def check(stagehand, page, browser):
            await page.goto(f"{site}/login.html")
            result = await stagehand.extract("is the user signed in, and what does the page say", SignedIn)
            return result.data

        await with_stagehand(sign_in, user_data_dir=profile)
        return await with_stagehand(check, user_data_dir=profile)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


SCENARIO = Scenario(
    name="persisted_login",
    title="observe selectors, fill credentials, reuse the profile",
    source="cookbooks/persisted-login",
    look_for="2 traces; observe returns selectors; the password locator.fill shows [redacted], the email does not",
    run=run,
    expect=Expect(traces=2, forbidden=(PASSWORD,), spans_include=("stagehand.observe", "locator.fill")),
)

if __name__ == "__main__":
    main(SCENARIO)
