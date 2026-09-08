API Docs
=========

The backend exposes its API through FastAPI, which generates interactive
documentation automatically. Once the server is running (see
:doc:`running`), it is available at:

- `<http://localhost:8000/docs>`_ — Swagger UI, lets you try out endpoints
  directly from the browser.
- `<http://localhost:8000/redoc>`_ — ReDoc, a more readable static
  reference.

Both are generated from the same OpenAPI schema and are always in sync
with the running server's code. There is nothing to build or keep up to
date manually.

See :doc:`concepts/architecture` to learn how the API fits into the rest of the system.
