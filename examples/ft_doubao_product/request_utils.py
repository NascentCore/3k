# request_utils.py
import requests
from base_utils import hmac_sha256, hash_sha256, norm_query

def request(method, date, query, header, ak, sk, body, service, region, host, path, content_type):
    credential = {
        "access_key_id": ak,
        "secret_access_key": sk,
        "service": service,
        "region": region,
    }
    request_param = {
        "body": body if body else "",
        "host": host,
        "path": path,
        "method": method,
        "content_type": content_type,
        "date": date,
        "query": query,
    }

    x_date = request_param["date"].strftime("%Y%m%dT%H%M%SZ")
    short_x_date = x_date[:8]
    x_content_sha256 = hash_sha256(request_param["body"])
    sign_result = {
        "Host": request_param["host"],
        "X-Content-Sha256": x_content_sha256,
        "X-Date": x_date,
        "Content-Type": request_param["content_type"],
    }
    signed_headers_str = ";".join(["content-type", "host", "x-content-sha256", "x-date"])
    canonical_request_str = "\n".join([
        request_param["method"].upper(),
        request_param["path"],
        norm_query(request_param["query"]),
        "\n".join([
            "content-type:" + request_param["content_type"],
            "host:" + request_param["host"],
            "x-content-sha256:" + x_content_sha256,
            "x-date:" + x_date,
        ]),
        "",
        signed_headers_str,
        x_content_sha256,
    ])
    hashed_canonical_request = hash_sha256(canonical_request_str)
    credential_scope = "/".join([short_x_date, credential["region"], credential["service"], "request"])
    string_to_sign = "\n".join(["HMAC-SHA256", x_date, credential_scope, hashed_canonical_request])
    k_date = hmac_sha256(credential["secret_access_key"].encode("utf-8"), short_x_date)
    k_region = hmac_sha256(k_date, credential["region"])
    k_service = hmac_sha256(k_region, credential["service"])
    k_signing = hmac_sha256(k_service, "request")
    signature = hmac_sha256(k_signing, string_to_sign).hex()

    sign_result["Authorization"] = "HMAC-SHA256 Credential={}, SignedHeaders={}, Signature={}".format(
        credential["access_key_id"] + "/" + credential_scope,
        signed_headers_str,
        signature,
    )
    header.update(sign_result)
    url = "https://{}{}".format(request_param["host"], request_param["path"])
    response = requests.request(
        method=method,
        url=url,
        headers=header,
        params=request_param["query"],
        data=request_param["body"],
        stream=True,
    )
    return response

