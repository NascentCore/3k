# base_utils.py
import hashlib
import hmac
from urllib.parse import quote

def norm_query(params):
    query = ""
    for key in sorted(params.keys()):
        if isinstance(params[key], list):
            for k in params[key]:
                query += quote(key, safe="-_.~") + "=" + quote(k, safe="-_.~") + "&"
        else:
            query += quote(key, safe="-_.~") + "=" + quote(params[key], safe="-_.~") + "&"
    return query[:-1].replace("+", "%20")

def hmac_sha256(key: bytes, content: str):
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()

def hash_sha256(content: str):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

