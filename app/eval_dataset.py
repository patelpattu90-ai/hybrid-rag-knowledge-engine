eval_queries = [

    # === index.md (FastAPI intro) ===
    {"query": "What is FastAPI?", "relevant_sections": ["FastAPI"]},
    {"query": "Why is FastAPI fast?", "relevant_sections": ["Requirements", "FastAPI"]},
    {"query": "What are FastAPI requirements?", "relevant_sections": ["Requirements"]},
    {"query": "How to install FastAPI?", "relevant_sections": ["Installation"]},
    {"query": "FastAPI example upgrade", "relevant_sections": ["Example upgrade"]},
    {"query": "Why do developers like FastAPI?", "relevant_sections": ["FastAPI"]},
    {"query": "What makes FastAPI powerful?", "relevant_sections": ["FastAPI"]},

    # === first-steps.md ===
    {"query": "How to create a FastAPI app?", "relevant_sections": ["First Steps"]},
    {"query": "What is the simplest FastAPI example?", "relevant_sections": ["First Steps"]},
    {"query": "How to run a FastAPI server?", "relevant_sections": ["First Steps"]},
    {"query": "What does FastAPI return by default?", "relevant_sections": ["First Steps"]},
    {"query": "How to use uvicorn with FastAPI?", "relevant_sections": ["First Steps"]},

    # === dependencies/index.md ===
    {"query": "What is dependency injection?", "relevant_sections": ["What is \"Dependency Injection\"", "Dependency Injection"]},
    {"query": "How does dependency injection work in FastAPI?", "relevant_sections": ["Dependency Injection", "Simple usage"]},
    {"query": "How to use dependencies in FastAPI?", "relevant_sections": ["Simple usage"]},
    {"query": "How are dependencies declared in FastAPI?", "relevant_sections": ["Dependency Injection"]},
    {"query": "How to build reusable components in FastAPI?", "relevant_sections": ["Dependency Injection"]},
    {"query": "What is the purpose of dependencies in FastAPI?", "relevant_sections": ["Dependency Injection"]},
    {"query": "Explain dependency injection example in FastAPI", "relevant_sections": ["Simple usage"]},
    {"query": "How to share logic using dependencies?", "relevant_sections": ["Dependency Injection", "Simple usage"]},
    {"query": "What is FastAPI dependency architecture?", "relevant_sections": ["Dependency Injection"]},
    {"query": "Explain FastAPI dependency system", "relevant_sections": ["Dependency Injection"]},

    # === path-params.md ===
    {"query": "What are path parameters in FastAPI?", "relevant_sections": ["Path Parameters"]},
    {"query": "How to declare path parameters in FastAPI?", "relevant_sections": ["Path Parameters"]},
    {"query": "How to use path parameters with types?", "relevant_sections": ["Path parameters with types"]},
    {"query": "What is data validation in FastAPI path params?", "relevant_sections": ["Data Validation"]},
    {"query": "How does FastAPI handle predefined path values?", "relevant_sections": ["Predefined values"]},
    {"query": "What are numeric path parameter constraints?", "relevant_sections": ["Number validations: greater than or equal", "Number validations: greater than and less than or equal"]},

    # === query-params.md ===
    {"query": "What are query parameters in FastAPI?", "relevant_sections": ["Query Parameters"]},
    {"query": "How to declare optional query parameters?", "relevant_sections": ["Optional parameters"]},
    {"query": "How to set default values for query parameters?", "relevant_sections": ["Query Parameters"]},
    {"query": "How to use multiple path and query parameters?", "relevant_sections": ["Multiple path and query parameters"]},
    {"query": "How to make query parameters required in FastAPI?", "relevant_sections": ["Required query parameters"]},
    {"query": "How to use boolean query parameters in FastAPI?", "relevant_sections": ["Query parameter type conversion"]},

    # === body.md ===
    {"query": "How to send a request body in FastAPI?", "relevant_sections": ["Request Body"]},
    {"query": "How to define a Pydantic model in FastAPI?", "relevant_sections": ["Import Pydantic's BaseModel", "Create your data model"]},
    {"query": "How to use request body with path parameters?", "relevant_sections": ["Request body + path parameters"]},
    {"query": "How to use request body with query parameters?", "relevant_sections": ["Request body + path + query parameters"]},
    {"query": "What is a request body in FastAPI?", "relevant_sections": ["Request Body"]},

    # === middleware.md ===
    {"query": "What is middleware in FastAPI?", "relevant_sections": ["Middleware"]},
    {"query": "How to create middleware in FastAPI?", "relevant_sections": ["Create a middleware"]},
    {"query": "How to add middleware to FastAPI?", "relevant_sections": ["Middleware"]},
    {"query": "What does middleware do in FastAPI?", "relevant_sections": ["Middleware"]},
    {"query": "How to measure request time using middleware?", "relevant_sections": ["Create a middleware"]},
]