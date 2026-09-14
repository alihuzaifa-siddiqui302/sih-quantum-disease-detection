# Clinical Decision Support API

> **Production Location Note:**  
> The production FastAPI clinical decision backend service is located at:
> - **FastAPI Application:** [`platform/backend/main.py`](file:///c:/Users/ADMIN/Desktop/sih-quantum/platform/backend/main.py)
> - **Automated Integration Tests:** [`tests/test_integration_backend.py`](file:///c:/Users/ADMIN/Desktop/sih-quantum/tests/test_integration_backend.py)

To start the API service locally:
```bash
uvicorn platform.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is served at:
`http://127.0.0.1:8000/docs`
