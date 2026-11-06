import test from "node:test";
import assert from "node:assert/strict";

import { context, trace } from "@opentelemetry/api";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { RespanSpanAttributes } from "@respan/respan-sdk";

import {
  getClient,
  updateCurrentSpan,
  withRespanSpanAttributes,
} from "../dist/index.js";
import { TIME_TO_FIRST_TOKEN_ATTR } from "../dist/constants/index.js";

const CUSTOMER_ID_ATTR = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID;
const VALID_SECONDS = 0.42;
const INVALID_VALUES = [true, false, -0.1, Infinity, -Infinity, NaN, "abc", "0.42", null, [0.42]];

// register() installs the async context manager the span helpers read from.
const provider = new NodeTracerProvider();
provider.register();
const tracer = provider.getTracer("respan-params-ttft-test");

test("local key mirrors the @respan/respan-sdk constant", () => {
  assert.equal(TIME_TO_FIRST_TOKEN_ATTR, RespanSpanAttributes.TIME_TO_FIRST_TOKEN);
});

async function attributesAfter(setParams) {
  const span = tracer.startSpan("ttft_span");
  await context.with(trace.setSpan(context.active(), span), setParams);
  span.end();
  return span.attributes;
}

const SNAKE_CASE_SETTERS = {
  updateCurrentSpan: (params) => async () => {
    updateCurrentSpan({ respanParams: params });
  },
  withRespanSpanAttributes: (params) => () =>
    withRespanSpanAttributes(() => undefined, params),
};

for (const [setterName, setter] of Object.entries(SNAKE_CASE_SETTERS)) {
  for (const key of ["time_to_first_token", "ttft"]) {
    test(`${setterName}: ${key} sets the bare time_to_first_token attribute`, async () => {
      const attrs = await attributesAfter(setter({ [key]: VALID_SECONDS }));
      assert.equal(attrs[TIME_TO_FIRST_TOKEN_ATTR], VALID_SECONDS);
    });
  }

  test(`${setterName}: zero is kept`, async () => {
    const attrs = await attributesAfter(setter({ ttft: 0 }));
    assert.equal(attrs[TIME_TO_FIRST_TOKEN_ATTR], 0);
  });

  for (const value of INVALID_VALUES) {
    test(`${setterName}: invalid ${JSON.stringify(value)} is not set, other params survive`, async () => {
      const attrs = await attributesAfter(
        setter({ time_to_first_token: value, customer_identifier: "user-123" })
      );
      assert.equal(TIME_TO_FIRST_TOKEN_ATTR in attrs, false);
      assert.equal(attrs[CUSTOMER_ID_ATTR], "user-123");
    });
  }
}

for (const key of ["timeToFirstToken", "ttft"]) {
  test(`getClient().updateCurrentSpan: ${key} sets the bare time_to_first_token attribute`, async () => {
    const attrs = await attributesAfter(async () => {
      getClient().updateCurrentSpan({ respanParams: { [key]: VALID_SECONDS } });
    });
    assert.equal(attrs[TIME_TO_FIRST_TOKEN_ATTR], VALID_SECONDS);
  });
}

for (const value of INVALID_VALUES) {
  test(`getClient().updateCurrentSpan: invalid ${JSON.stringify(value)} is not set`, async () => {
    const attrs = await attributesAfter(async () => {
      getClient().updateCurrentSpan({
        respanParams: { timeToFirstToken: value, customerIdentifier: "user-123" },
      });
    });
    assert.equal(TIME_TO_FIRST_TOKEN_ATTR in attrs, false);
    assert.equal(attrs[CUSTOMER_ID_ATTR], "user-123");
  });
}
