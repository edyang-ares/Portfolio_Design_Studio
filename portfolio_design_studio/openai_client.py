import os
import ssl
import truststore
import httpx
from openai import OpenAI

def make_openai_client() -> OpenAI:
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    http_client = httpx.Client(verify=ctx)

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=http_client,
    )