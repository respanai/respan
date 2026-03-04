# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking

- **`RespanChatGenerator` and `RespanGenerator`: `model` is no longer required and has no default.**  
  You must provide **either** `model` or `prompt_id` when constructing the generator. If you previously relied on the default `model="gpt-3.5-turbo"`, you will now get a `ValueError`. **Migration:** pass `model` explicitly, e.g. `RespanChatGenerator(model="gpt-3.5-turbo", ...)` or use a platform-managed prompt with `prompt_id`.

- **`RespanGenerator`: `streaming_callback` has been removed.**  
  Passing `streaming_callback` to the constructor will raise a `TypeError`. This is part of the revamp; streaming behavior is no longer supported via this parameter. Remove any `streaming_callback` arguments when upgrading.

### Security

- **`to_dict` no longer serializes the API key.**  
  `RespanConnector` and `_BaseRespanGenerator` (used by `RespanGenerator` and `RespanChatGenerator`) no longer include the API key in serialized output, so saved pipelines or logs do not leak secrets. When loading a pipeline, the key is resolved from the `RESPAN_API_KEY` environment variable.

## [1.0.0]

- Initial release.
