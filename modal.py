import pathlib

import modal

with open("requirements.txt") as f:
    r = f.read()

packages_list = r.split("\n")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "aiohttp==3.8.5",
        "fastapi==0.109.2",
        "langchain==0.1.5",
        "nltk==3.8.1",
        "numpy==1.25.0",
        "openai==0.27.8",
        "pandas==2.0.3",
        "# pydantic==1.10.12",
        "python-dotenv==1.0.1",
        "qdrant_client==1.7.3",
        "ratelimit==2.2.1",
        "Requests==2.31.0",
        "scikit_learn==1.3.0",
        "sentence_transformers==2.3.1",
        "starlette==0.36.3",
        "streamlit==1.24.1",
        "tenacity==8.2.2",
        "torch==2.0.1",
        "tqdm==4.65.0",
        "typing_extensions==4.9.0",
        "unstructured==0.8.1",
        "uvicorn",
        "streamlit-feedback==0.1.3",
    )
    # Use fork until https://github.com/valohai/asgiproxy/pull/11 is merged.
    .pip_install("git+https://github.com/modal-labs/asgiproxy.git")
)

stub = modal.Stub(name="Financial-Docs-LLM", image=image)

streamlit_script_local_path = pathlib.Path(__file__).parent / "Intro.py"
streamlit_script_remote_path = pathlib.Path("/root/Intro.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "Intro.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

HOST = "127.0.0.1"
PORT = "8000"


def spawn_server():
    import socket
    import subprocess

    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(streamlit_script_remote_path),
            "--browser.serverAddress",
            HOST,
            "--server.port",
            PORT,
            "--browser.serverPort",
            PORT,
            "--server.enableCORS",
            "false",
        ]
    )

    # Poll until webserver accepts connections before running inputs.
    while True:
        try:
            socket.create_connection((HOST, int(PORT)), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


@stub.function(
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
)
@modal.asgi_app()
def run():
    from asgiproxy.config import BaseURLProxyConfigMixin, ProxyConfig
    from asgiproxy.context import ProxyContext
    from asgiproxy.simple_proxy import make_simple_proxy_app

    spawn_server()

    config = type(
        "Config",
        (BaseURLProxyConfigMixin, ProxyConfig),
        {
            "upstream_base_url": f"http://{HOST}:{PORT}",
            "rewrite_host_header": f"{HOST}:{PORT}",
        },
    )()
    proxy_context = ProxyContext(config)
    return make_simple_proxy_app(proxy_context)
